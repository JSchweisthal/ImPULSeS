# ImPULSeS
Implementation of ImPULSeS: Robust and Efficient Imbalanced Positive-Unlabeled
Learning with Self-supervision

<p align="center">
  <img src="media/flowchart.png?raw=true" width="500"/>
</p>

Create conda environment for impulses
```
conda env create -f environment.yml
conda activate impulses
```

Do pretraining (options can be modified in config/temp.yaml)
```
python main_pretrain.py -c temp
```
Afterwards: finetuning of classification head (options can be modified in config/temp.yaml)
```
python main_finetune.py -c temp
```
Default setting of config/temp.yaml: Pre-training with debiased contrastive loss and fine-tuning with imbalanced nnPU loss both on imbalanced CIFAR10 data.

#
Parts of Code used from: 
1. https://github.com/spijkervet/SimCLR
2. https://github.com/chingyaoc/DCL
3. https://github.com/guangxinsuuu/Positive-and-Unlabeled-Learning-from-Imbalanced-Data 

