# Adapted from tensorflow's tensorflow.contrib.learn.python.learn.datasets.mnist file
import numpy as np
from collections import namedtuple
from tensorflow.contrib.keras import datasets

KERAS_DATASETS = ['mnist', 'fashion_mnist']
# KERAS_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']  # TODO: color image datasets
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])


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


def load_keras_dataset(dataset='mnist', dtype=np.float32, reshape=True, validation_size=0, seed=123):
    # more clever way of handling this?
    if dataset == 'mnist':
        # keras provides datasets as tuples of numpy arrays of data type uint8
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    # elif dataset == 'fashion_mnist':
    #     (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()  # can't find?
    # elif dataset == 'cifar10':
    #     (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # elif dataset == 'cifar100':
    #     (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    else:
        raise ValueError(
            'Unavailable dataset specified. Datasets available: [{}]'.format(', '.join(KERAS_DATASETS)))

    # prevent compatibility issues
    train_images = np.expand_dims(train_images, axis=-1)
    train_labels = np.expand_dims(train_labels, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    test_labels = np.expand_dims(test_labels, axis=-1)

    if not 0 <= validation_size <= train_images.shape[0]:
        raise ValueError('Validation size should be between 0 and {}. Received {}.'
                         .format(train_images.shape[0], validation_size))

    # no point in validation set here?
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = Dataset(train_images, train_labels, **options)
    validation = Dataset(validation_images, validation_labels, **options)
    test = Dataset(test_images, test_labels, **options)

    return Datasets(train=train, validation=validation, test=test)


# TODO: implement ability to load frey face dataset
# TODO: implement ability to load celebA dataset


# for debugging
if __name__ == "__main__":
    dataset = load_keras_dataset()
    print(dataset.train.num_examples)
