
from config.miscdata_small_pyz_finetune_simclr_nodist import Config

RandomUpSamplerEpochTotal = 2050

Config.data_params['sampler_labelled'] = 'RandomUpSampler'
Config.data_params['sampler_conf'] = {'shuffle': True, 'upsample_to': RandomUpSamplerEpochTotal, 'seed':10}
Config.data_params['labelled_samples'] = 0

Config.data_params['selected_labels'] = {
            'samples': 40,
            'policy': 'load_indices_vector', # coreset_k_centres_greedy # random_label_balanced
            'load_indices_csv': 'indices/cifar10_random_labelled.csv'
        }

Config.experiment_params['use_head_proj'] = 0
Config.run_name = 'cifar10_semisupervised_finetune'

Config.early_stopping_patience = 5
