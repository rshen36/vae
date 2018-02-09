# Adapted from tensorflow's tensorflow.contrib.learn.python.learn.datasets.mnist file
import numpy as np
from tensorflow.contrib.keras import datasets

KERAS_DATASETS = ['mnist', 'fashion_mnist']
# KERAS_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']  # TODO: color image datasets


class Dataset:
    def __init__(self,
                 images,
                 labels,
                 dtype=np.float32,
                 reshape=True,
                 seed=123):
        """
        `dtype` should be either `uint8` to leave input as `[0, 255]` or `float32` to rescale into `[0, 1]`
        """
        np.random.seed(seed)
        if dtype not in (np.uint8, np.float32):
            raise TypeError(
                'Invalid image dtype {}, expected uint8 or float32'.format(dtype))
        assert images.shape[0] == labels.shape[0], (
            'images.shape: {} labels.shape: {}'.format(images.shape, labels.shape))
        assert type(seed) is int, (
            'Invalid seed specified: {}'.format(seed))
        self._num_examples = images.shape[0]

        # flatten images
        # TODO: adjust for color images
        if reshape:
            assert len(images.shape) == 3 or images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == np.float32:
            # convert from [0, 255] --> [0.0, 1.0]
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        if start + batch_size > self._num_examples:
            # finished epoch
            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self.labels[start:end]


def load_keras_dataset(dataset='mnist'):
    # more clever way of handling this?
    if dataset == 'mnist':
        # keras provides datasets as tuples of numpy arrays (data type?)
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # elif dataset == 'fashion_mnist':
    #     (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()  # can't find?
    # elif dataset == 'cifar10':
    #     (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # elif dataset == 'cifar100':
    #     (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    else:
        raise ValueError(
            'Unavailable dataset specified. Datasets available: [{}]'.format(', '.join(KERAS_DATASETS)))


# TODO: implement ability to load frey face dataset
# TODO: implement ability to load celebA dataset


# for debugging
if __name__ == "__main__":
    dataset = load_keras_dataset()
    print(dataset.train.num_examples)
