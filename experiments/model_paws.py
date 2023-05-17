# https://github.com/kekmodel/FixMatch-pytorch

import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.loss import init_paws_loss, make_labels_matrix

from src.utils import PredictionCache, expand_inputs, log_sum_exp
from src.utils import _ECELoss
from src.data import extract_subset_dataset
from torch.distributions.normal import Normal
from torch.optim import lr_scheduler

from experiments.model_base_ss import ModelBaseSemiSupervisedCombined
import numpy as np
import math

from pdb import set_trace as pb

import time
from argparse import Namespace
from torchvision import transforms
import copy
import torch.distributed as dist

class PAWS(ModelBaseSemiSupervisedCombined):
    def __init__(self, config):
        super(PAWS, self).__init__(config)
        conf = Namespace(**self.config.experiment_params)
        # =======================================================================
        
        if hasattr(conf, 'model_class'):
            model = conf.model_class.load_from_checkpoint(conf.model_path)
            self.model = model
            self.model_path = conf.model_path
            if hasattr(self.model, 'set_head_projection'):
                self.model.set_head_projection(conf.use_head_proj)
        else:
            self.model = conf.model
        
        self.classes = conf.classes
        # self.encoder.requires_grad = False
        
        self.freeze_encoder = conf.freeze_encoder
        self.projection_head = conf.projection_head

        if hasattr(conf, 'batchnorm_momentum'):
            for m in self.modules():
                # set batchnorm momentum
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    m.momentum = conf.batchnorm_momentum
        
        # self.automatic_optimization=False
        # ===============================
        self.label_smoothing = conf.label_smoothing
        self.sharpen = conf.sharpen
        self.me_max = conf.me_max

        self.temperature = conf.temperature
        # ===============================

        # ===============================
        self.ece_criterion = _ECELoss()
        # ===============================
        self.active_learning_save_representations = True
        self.evaluate_labelled = True
        self.z_points = 1

        # ===============================
        # if (hasattr(conf, 'representation_output')):
        #     self.representation_output = conf.representation_output
# =============================================================================================

    def configure_optimizers(self):
        self.setup_init_data()

        self.trainer.datamodule.dataset_labelled.dataset.supervised = True
        self.effective_classes = self.trainer.datamodule.dataset_labelled.dataset.set_supervised(self.trainer.datamodule.dataset_labelled.indices)

        # remove_subset = extract_subset_dataset(self.trainer.datamodule.dataset_labelled, obj_data = 'data', obj_target = 'targets')
        # self.trainer.datamodule.dataset_labelled.dataset = remove_subset
        # self.trainer.datamodule.dataset_labelled.indices = np.arange(len())

        self.trainer.datamodule.dataset_unlabelled.dataset.supervised = False

        if self.effective_classes != len(self.trainer.datamodule.dataset_labelled.dataset.classes):
            self.trainer.datamodule.sampler_conf['classes_per_batch'] = self.effective_classes 
            self.trainer.datamodule.dataset_labelled.dataset.classes = np.arange(self.effective_classes)

        self.unique_classes= self.trainer.datamodule.sampler_conf['unique_classes']
        self.classes_per_batch= self.trainer.datamodule.sampler_conf['classes_per_batch']

        self.multicrop = self.trainer.datamodule.dataset_labelled.dataset.multicrop_transform[0]
        self.supervised_views = self.trainer.datamodule.dataset_labelled.dataset.supervised_views


        self.batch_size = self.trainer.datamodule.batch_size
        self.s_batch_size = self.batch_size//self.effective_classes #self.classes
        self.u_batch_size = int(self.batch_size*self.trainer.datamodule.batch_size_unlabelled_scalar)

        if torch.distributed.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.paws = init_paws_loss(
                multicrop=self.multicrop,
                tau=self.temperature,
                T=self.sharpen,
                me_max=self.me_max
            )

        self.labels_matrix = make_labels_matrix(
                num_classes=self.effective_classes,
                s_batch_size=self.s_batch_size,
                world_size=self.world_size,
                device=self.device,
                unique_classes=self.unique_classes,
                smoothing=self.label_smoothing
            )

        self.trainer.reset_train_dataloader(self)
        loaders = self.trainer.train_dataloader.loaders

        if (hasattr(loaders['labelled'], 'loader')):
            tmp = math.ceil(len(loaders['unlabelled']) / len(loaders['labelled'].loader))
            loaders['labelled'].loader.batch_sampler.set_inner_epochs(tmp)
        else:
            tmp = math.ceil(len(loaders['unlabelled']) / len(loaders['labelled']))
            loaders['labelled'].batch_sampler.set_inner_epochs(tmp)

        self.trainer.reset_train_dataloader(self)
        loaders = self.trainer.train_dataloader.loaders
        
        return self.setup_optimizers()

# =============================================================================================

    def _evaluate_z(self, x, grad=None):
        if hasattr(self, 'model_path'):
            z, z_p = self.model(x, grad=grad, gen=False)
        else:
            z = self.model(x)
            z_p = None
        return z, z_p

    def evaluate_z(self, x, grad=None):
        if grad is None:
            z, z_p = self._evaluate_z(x)
        else:
            with torch.set_grad_enabled(grad):
                if not grad:
                    self.model.eval()
                z, z_p = self._evaluate_z(x)
        return z, z_p

# =============================================================================================
    def evaluate_image_list(self, inputs, return_before_head=False, grad=None):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _h, _ = self.evaluate_z(torch.cat(inputs[start_idx:end_idx]), grad = grad)
            _h = self.projection_head(_h)
            _z = _h
            if start_idx == 0:
                h, z = _h, _z
            else:
                h, z = torch.cat((h, _h)), torch.cat((z, _z))
            start_idx = end_idx

        if return_before_head:
            return h, z

        return z


    def full_step(self, batch_labelled, batch_unlabelled, total_batches, validate=True):
        
        return_prob_labelled = {}
        return_prob_unlabelled = {}
        if batch_unlabelled is None:
            x, y, idx = batch_labelled
            z = self.evaluate_image_list(x, return_before_head=False, grad=False)
            probs = self.snn(z)
            # print(probs)
            loss = 0
            return_prob_labelled = {'acc': probs, 'label': y, 'index':idx}
            # pb()
        else:
            # labels = batch_labelled[self.supervised_views]
            # start = time.time()
            labels = torch.cat([self.labels_matrix for _ in range(self.supervised_views)])
            imgs = batch_labelled[:-1] + batch_unlabelled[:-1]
            # end = time.time()
            # print('loadimg')
            # print(end-start)


            # start = time.time()
            h, z = self.evaluate_image_list(imgs, return_before_head=True, grad=not self.freeze_encoder)

            # Compute paws loss in full precision
            with torch.cuda.amp.autocast(enabled=False):

                # Step 1. convert representations to fp32
                if self.config.precision == 64:
                    h, z = h, z
                else:
                    h, z = h.float(), z.float()

                # Step 2. determine anchor views/supports and their
                #         corresponding target views/supports
                # --
                num_support = self.supervised_views * self.s_batch_size * self.classes_per_batch
                # --
                anchor_supports = z[:num_support]
                anchor_views = z[num_support:]
                # --
                target_supports = h[:num_support].detach()
                target_views = h[num_support:].detach()
                target_views = torch.cat([
                    target_views[self.u_batch_size:2*self.u_batch_size],
                    target_views[:self.u_batch_size]], dim=0)

                # Step 3. compute paws loss with me-max regularization
                (ploss, me_max) = self.paws(
                    anchor_views=anchor_views,
                    anchor_supports=anchor_supports,
                    anchor_support_labels=labels,
                    target_views=target_views,
                    target_supports=target_supports,
                    target_support_labels=labels)
                loss = ploss + me_max
                return_prob_unlabelled = {'ploss': ploss, 'me_max': me_max}
                if hasattr(self.config, 'test_function'):
                    self.config.test_function(self, ploss)

        if hasattr(self.config, 'pdb_debug'):
            if self.config.pdb_debug:
                torch.set_printoptions(precision=16)
                pb()

        unlabelled_loss = 0
        labelled_loss = loss
        # =========================================================================================
        return labelled_loss, return_prob_labelled, unlabelled_loss, return_prob_unlabelled

# ===============================================================================


    def predict_step(self, batch, batch_idx):
        x = batch[0] # first supervised view
        y = batch[-1] # last column is labels
        z = self.evaluate_image_list(x, return_before_head=False, grad=None)
        # if batch_idx == 1:
        # print(x[0])
        # print(y[0])
        # print(z[0])

        return z, y

    def make_snn(self, embs, labs, temp=0.1):

        # --Normalize embeddings
        embs = embs.div(embs.norm(dim=1).unsqueeze(1)).detach_()
        # print(embs)

        softmax = torch.nn.Softmax(dim=1)

        def snn(h, h_train=embs, h_labs=labs):
            # -- normalize embeddings
            h = h.div(h.norm(dim=1).unsqueeze(1))
            return softmax(h @ h_train.T / temp) @ h_labs
        return snn

    def on_validation_epoch_start(self):
        self.trainer.predict_loop.epoch_loop.return_predictions = True
        self.trainer.predict_loop._return_predictions: bool = True

        datamodule = self.trainer.datamodule
        datamodule.predict_dataset = 'labelled'
        datamodule.use_validation_transforms_for_predict = True
        self.trainer.reset_predict_dataloader(self)
        predictions = self.trainer.predict_loop.run()

        embs = [x[0] for x in predictions]
        labs = [x[1] for x in predictions]
        
        labs = torch.cat(labs)
        labs = labs.long().view(-1, 1)
        labs = torch.full((labs.size()[0], self.classes), 0.).scatter_(1, labs, 1.)
        if self.config.precision == 64:
            labs = labs.double()

        self.embs = torch.cat(embs).to(self.device)
        self.labs = labs.to(self.device)
        temp=0.1
        self.snn = self.make_snn(self.embs, self.labs, temp)

# ==============================================================

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_validation_epoch_end(self):
        del self.embs
        del self.labs
        del self.snn
        self.embs = None
        self.labs = None
        self.snn = None

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()