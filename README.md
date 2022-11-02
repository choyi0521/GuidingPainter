# GuidingPainter

Official implementation of “Guiding Users to Where to Give Color Hints for Efficient Interactive Sketch Colorization via Unsupervised Region Prioritization” (WACV 2023)

## Prerequisites
* pytorch
* tensorboard
* pytorch lightning
* albumentations

## Important options
* phase: train or test
* processed_dir: directory which contains datasets
* train_dir: drectory in which checkpoints and logs are saved
* test_dir: drectory in which test results are saved
* dataset: the name of dataset
* gpus: gpus which are used for training
* test_gpu: single gpu which is used for testing
* checkpoint: filename of checkpoint

## Training example
```
python main.py \
--dataset yumi \
--processed_dir {processed dir} \
--train_dir ../save/train \
--test_dir ../save/test \
--phase train \
--epochs 200 \
--batch_size 12 \
--gpus 0 \
--test_gpu 0
```

## Expected dataset structure
processed_dir  
&nbsp;&nbsp;&nbsp;&nbsp; ㄴ{dataset name}  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴtrain: used for training  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴcolor: directory which contains color images  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴsketch: directory which contains sketch images  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴval: used for validation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴcolor  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴsketch  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴtest: used for testing  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴcolor  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ㄴsketch  
