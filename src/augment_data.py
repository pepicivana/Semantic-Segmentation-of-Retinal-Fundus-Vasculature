import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from skimage.io import imread
from scipy.misc import imsave
from skimage.transform import PiecewiseAffineTransform, warp
import scipy
import numpy as np
import os


def augment_data(args):
    print('Augmenting image data sets ...')
    raw_images_dir = os.path.join(args.raw_image_directory, 'images')
    raw_annotations_dir = os.path.join(args.raw_image_directory, 'annotations')

    augmented_images_path = os.path.join(args.augmented_image_directory, args.augmented_image_filename)

    images = []
    annotations = []
    image_list = os.listdir(raw_images_dir)

    for image in image_list:
        img = imread(os.path.join(raw_images_dir, image), as_grey=True).astype(np.float32)
        img_segmented = imread(os.path.join(raw_annotations_dir, image[:-4] + '.tif')).astype(np.float32)

        img = scipy.misc.imresize(img, (512, 512))
        img_segmented = scipy.misc.imresize(img_segmented, (512, 512))

        img = normalize_image(img)
        img_segmented = normalize_image(img_segmented)

        images.append(img)
        annotations.append(img_segmented)


    # split images and annotations into train/test
    X_train, y_train, X_test, y_test = train_val_test_split(np.array(images, dtype=np.float32),
                                                            np.array(annotations, dtype=np.float32))
    # # send to augment function one at a time
    X_train, y_train = augment_images(X_train, y_train)
    X_test, y_test = augment_images(X_test, y_test)
    #
    # # save to npz
    np.savez(augmented_images_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def augment_images(img_array, img_segmented_array):
    images_augmented = []
    annotations_augmented = []

    for img, img_segmented in zip(img_array, img_segmented_array):

        img_lr, img_ud = flip_transform_image(img)

        img_segmented_lr, img_segmented_ud = flip_transform_image(img_segmented)

        img_warped, img_segmented_warped = non_linear_warp_transform(img, img_segmented)

        img_lr_warped, img_segmented_lr_warped = non_linear_warp_transform(img_lr, img_segmented_lr)

        img_ud_warped, img_segmented_ud_warped = non_linear_warp_transform(img_ud, img_segmented_ud)

        images_augmented.extend([img, img_lr, img_ud, img_warped, img_lr_warped, img_ud_warped])
        annotations_augmented.extend([img_segmented, img_segmented_lr, img_segmented_ud,
                                      img_segmented_warped, img_segmented_lr_warped, img_segmented_ud_warped])

    images_augmented = np.array(images_augmented, dtype=np.float32)[:, :, :, np.newaxis]
    annotations_augmented = np.array(annotations_augmented, dtype=np.float32)[:, :, :, np.newaxis]

    return (images_augmented, annotations_augmented)


def normalize_image(img):
    return ((img - img.min()) / (img.max() - img.min()))


def train_val_test_split(images, annotations):
    p = np.random.permutation(images.shape[0])
    images, annotations = images[p, :, :], annotations[p, :, :]
    samples = images.shape[0]
    X_train = images[0:int(0.8 * samples), :, :]
    y_train = annotations[0:int(0.8 * samples), :, :]
    # X_validate = images[int(0.6 * samples):int(0.8 * samples), :, :]
    # y_validate = annotations[int(0.6 * samples):int(0.8 * samples), :, :]
    X_test = images[int(0.8 * samples):, :, :]
    y_test = annotations[int(0.8 * samples):, :, :]
    return X_train, y_train, X_test, y_test


def non_linear_warp_transform(img, annotation):
    rows, cols = img.shape[0], img.shape[1]

    src_cols = np.linspace(0, cols, 6)
    src_rows = np.linspace(0, rows, 6)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    dst = np.random.normal(0.0, 10, size=(36, 2)) + src

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = img.shape[0]
    out_cols = img.shape[1]
    img_out = warp(img, tform, output_shape=(out_rows, out_cols))
    annotation_out = warp(annotation, tform, output_shape=(out_rows, out_cols))
    return img_out, annotation_out


def flip_transform_image(img):
    img_lr = np.fliplr(img)
    img_ud = np.flipud(img)
    return img_lr, img_ud
