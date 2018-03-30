# U-Net Vessel Segmentation

![U-Net output](https://github.com/b-schneller/U-Net-Vessel-Segmentation/blob/master/unet.png)

## Introduction

Semantic segmentation technqiues attempt to label every pixel in an input image to belonging to a set of n classes. This code implements the [U-Net](https://arxiv.org/abs/1505.04597) architecture developed by Ronneberger, Fischer and  Brox for use in segmentating anatomical regions in medical images. In this implemention the network is trained and evaluated on [High-Resolution Funcdus Images](https://www5.cs.fau.de/research/data/fundus-images/) compiled for use by the Friedrich-Alexander-Universität Erlangen-Nürnberg. After training the model was able to achieve a Dice coefficient of 0.92 on the test data set.

## To Run

Clone the repo, and move to the main directory.
`cd ~/U-Net-Vessel-Segmentation/src`
Install the dependencies.
`$ pip install -r requirements.txt`
If running for the first time run:
`python main.py --make_dataset` and optionally `--augment_data`

The operation `--make_dataset` creates the data directory structure and downloads the image files and their expert labeled annotations. 

Data augmentation, which is highly recommended, includes up/down and left/right reflection as well as random non-linear warping of the raw images. After augmentation number of samples increases from 45 to 270. For each input image the the same transformations are applied to the annotated images for use in training and evaluation. After this process the transformed images are saved, so this step only needs to be performed once. 


## To Train

`python main.py --train`
#### Optional settings include:
`--n_points`: Number of points from each cloud to sample for training/testing. Required for mini-batch training.
`--batch_size`: Batch size for training/testing
`--n_epochs`: Number of passes through training set.
`--early_stopping_max_checks`: Stop early when loss does not improve for max_checks.
`--learning_rate`: Learning rate for Adam Optimizer. This is the initial learning rate. The rate will is halved every 50 epochs.

## To Test

Testing requires saved model after training. 
To run test:
`python main.py --infer --load_checkpoint <saved_model_name>.ckpt`
The labeled output images will be saved in `~/U-Net-Vessel-Segmentation/data/output/`
