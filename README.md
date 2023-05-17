
# Cold PAWS: repository to reproduce SimCLR pretraining, finetuning and PAWS results

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

# Using labelled subsets

In the indices folder, we have some example input files for selecting small labelled subsets for training.

These can be finetuned with

```bash
python main.py --config config/miscdata_small_pyz_finetune_simclr_nodist_some.py 
```

These can be fit using paws with

```bash
python main.py --config config/miscdata_paws_small.py 
```

