
from config.miscdata_small_pyz_finetune_simclr_nodist import Config

RandomUpSamplerEpochTotal = 10000

Config.data_params['sampler_labelled'] = 'RandomUpSampler'
Config.data_params['sampler_conf'] = {'upsample_to': 1, 'seed':10}
Config.data_params['batch_size'] = 64
Config.data_params['labelled_samples'] = 0

Config.data_params['selected_labels'] = {
            'samples': 20,
            'policy': 'evaluate_representations',
            'active_learning_print_representations': True,
            'representation_output': 'all'
        }

Config.experiment_params['save_representation'] = ['index', 'label', 'encoder_mapping', 'encoder']

Config.max_epochs = 0
Config.run_name = 'cifar10_simclr_writedata'
