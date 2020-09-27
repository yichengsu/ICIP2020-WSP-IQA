# BLIND NATURAL IMAGE QUALITY PREDICTION USING CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING
By Yicheng Su(suyicheng1995@gmail.com) and Jari Korhonen

## Introduction
This repository is the [PyTorch](http://pytorch.org) implementation of "[BLIND NATURAL IMAGE QUALITY PREDICTION USING CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING](#)" in ICIP2020.

<p align = 'center'>
<img src = 'img/model.jpg' width = '627px'>
</p>
<p align = 'center'>
Illustration of the proposed weighted spatial pooling scheme
</p>


There are example images and corresponding weight maps for channels 500, 1000, 1500 and 2000.
(a) An example image from KonIQ-10k, and (b) respective weight maps.
(c) An example image from Live-itW, and (d)respective weight maps.


<p align = 'center'>
<img src = 'img/img1.jpg' height = '150px'>
<img src = 'img/img1_w.jpg' height = '150px'>
<img src = 'img/img2.jpg' height = '150'>
<img src = 'img/img2_w.jpg' height = '150'>
</p>


#### Citation

```
@inproceedings{su2020WSP,
  title={BLIND NATURAL IMAGE QUALITY PREDICTION USING CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING},
  author={Su, Yicheng and Korhonen, Jari},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  year={2020},
  organization={IEEE}
}
```

## Requirements
You will need the following requirements:
- numpy >= 1.17.4
- pandas >= 0.25.3
- python >= 3.7.5
- pytorch >= 1.4.0
- torchvision >= 0.5.0
- tensorboard >= 2.0.0

If you will reimplement in Linux, I highly recommend using the conda command line for installation:

    $ conda env create -f requirements.yml

## Dataset

The KonIQ-10k dataset has the directory structure as:

```
/path/to/koniq10k
  ├─1024x768
  │  ├─4042572733.jpg
  │  └─...
  └─koniq10k_scores_and_distributions.csv
```

And Live-itW has directory structure as following:

```
/path/to/LiveChallenge
  ├─Data
  │  ├─live_moc.csv
  │  └─...
  ├─Images
  │  ├─1000.JPG
  │  └─...
  └─README.txt
```

## Training
You can easily train the model using the command line:

    $ CUDA_VISIBLE_DEVICES=0 python main.py /path/to/koniq10k --tensorboard --comment VQANetTraing

`VQANetTraing` is for checkpoint name and tensorboard folder name.

## Evaluation
You can evaluate KonIQ-10k:

    $ CUDA_VISIBLE_DEVICES=0 python main.py /path/to/koniq10k -e -p -a checkpoint

You can also evaluate Live-itW:

    $ CUDA_VISIBLE_DEVICES=0 python cross_test.py /path/to/LiveChallenge/ checkpoint

`checkpoint` is in checkpoints folder and without suffix.

It's worth noting that if you training from scratch, I add a timestep in the front of checkpoint file and tensorboard, so the checkpoint name like `20200101-010101VQANetTraing`.

## Pre-trained Model
Download pre-trained model to checkpoints:

```
$ cd /path/to/ICIP2020-WSP-IQA
$ mkdir checkpoints
$ cd checkpoints
$ wget https://github.com/yichengsu/ICIP2020-WSP-IQA/releases/download/v0.1/checkpoint.pth.tar
$ cd ..
$ CUDA_VISIBLE_DEVICES=0 python main.py /path/to/koniq10k -e -p
```

You can use pre-trained model to get results as follow:

|           | PLCC  | SRCC  | RMSE  |
| :-------: | :---: | :---: | :---: |
| KonIQ-10k | 0.934 | 0.918 | 0.196 |
| Live-itW  | 0.849 | 0.825 | 0.720 |
