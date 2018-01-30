from make_dataset import make_dataset
from augment_data import augment_data
from model import Model
import argparse
import sys
import numpy as np
import os
import tensorflow as tf


def main(argv):
    ########### add directory creation ############

    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true', default=False,
                        help='Turn on to download data to disk.')
    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Turn on to augment raw data.')

    parser.add_argument('--raw_image_directory', default='../data/raw/',
                        help='Directory for downloaded images')
    parser.add_argument('--augmented_image_directory', default='../data/processed/',
                        help='Augmented image directory')
    parser.add_argument('--augmented_image_filename', default='augmented_images',
                        help='Augmented images filename')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of training epochs.')

    parser.add_argument('--saved_model_directory', default='../models/',
                        help='Directory for saving trained models')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Optimizer learning rate')
    parser.add_argument('--early_stopping_max_checks', type=int, default=20,
                        help='Max checks without improvement for early stopping')

    parser.add_argument('--train', action='store_true', default=False,
                        help='Set to True to train network')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Set to True to conduct inference on Test images. Trained model must be loaded.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Load saved checkpoint, arg=checkpoint_name')

    args = parser.parse_args()

    if not os.path.isdir(args.raw_image_directory):
        os.makedirs(args.raw_image_directory)
        os.makedirs(args.augmented_image_directory)
        os.makedirs(args.saved_model_directory)

    if args.download_data:
        make_dataset(args)

    if args.augment_data:
        augment_data(args)

    data = np.load(os.path.join(args.augmented_image_directory, args.augmented_image_filename + '.npz'))

    model = Model(args, data)
    model.build_net()

    if args.train:
        model.train()

    if args.infer and args.load_checkpoint is not None:
        model.infer()
    else:
        print('Trained model needs to be loaded for inference.')

if __name__ == '__main__':
    main(sys.argv)
