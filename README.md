# Cold PAWS: Unsupervised class discovery and addressing the cold-start problem for semi-supervised learning 
## Repository to reproduce SimCLR pretraining, finetuning and PAWS results

This is the pytorch code for fitting the models from the [paper](https://arxiv.org/abs/2305.10071). For the label selection strategy code see [this repo](https://github.com/emannix/cold-paws-labelling-selection-strategies).

---

To train the SimCLR model on CIFAR10, run

```bash
python main.py --config config/miscdata_paws_small.py 
```

For convenience, we have provided a CIFAR-10 pretrained model in the pretrained_models folder. You can download it from [here](https://drive.google.com/file/d/1yMV_hZtupooj9CiVVmVWPygGMZCxrJhQ/view?usp=share_link) along with some sample output files.

To train a supervised model from these pretrained weights, run

```bash
python main.py --config config/miscdata_small_pyz_finetune_simclr_nodist.py 
```

To write the SimCLR encodings to file for this pretrained model, run

```bash
python main.py --config config/miscdata_small_pyz_finetune_simclr_nodist_writedata.py
```

---
### Using particular labelled subsets

In the indices folder, we have some example input files for selecting small labelled subsets for training.

These can be finetuned with

```bash
python main.py --config config/miscdata_small_pyz_finetune_simclr_nodist_some.py 
```

These can be fit using paws with

```bash
python main.py --config config/miscdata_paws_small.py 
```

