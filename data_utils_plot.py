from skimage import color, transform
import imageio
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Queue, Process
import importlib
from copy import copy


def init(definitions_package):
    """
    initilize global variables imported from definitions
    """
    package = importlib.import_module(definitions_package)
    for object_name in package.__dict__:
        globals()[object_name] = getattr(package, object_name)
    global imported_definitions
    imported_definitions = definitions_package

    # data generation processes
    global source_files
    source_files = []
    for folder in source_folders:
        subfolder = os.path.join(root_dir, folder)
        tmp = [os.path.join(subfolder, file) for file in os.listdir(subfolder) if file.endswith('.jpg')]
        source_files.extend(tmp)
    print("Found %d files" % len(source_files))
    split = int(len(source_files) * 0.15)

    global min_value, max_value, delta
    min_value = -1
    max_value = +1
    delta = (max_value - min_value) / (BINS - 1)


### various helper function for one-hot-encoding
def encode_bin(img):
    img = np.clip(img.astype(np.float32), min_value, max_value)
    img = (img - min_value) / delta
    img = np.round(img).astype(np.uint8)
    return img


def encode_one_hot(img):
    binned_img = encode_bin(img)
    res = np.eye(BINS)[binned_img.reshape(-1)]
    res = res.reshape(list(img.shape) + [BINS])
    return res


def encode_one_hot_a(img):
    res = encode_one_hot(img)
    return res


def encode_one_hot_b(img):
    res = encode_one_hot(img)
    return res


def decode_bin(img):
    img = img.astype(np.float32) * delta + min_value
    return img


# main data generator
class ImageDataGenerator:
    def __init__(self, source_files, batch_size, target_size):
        self._source_files = copy(source_files)
        random.shuffle(self._source_files)
        self._batch_size = batch_size
        self._target_size = target_size
        self.__counter = 0
        self._batch_shape_1 = (batch_size, target_size, target_size, 1)

    def get_single_continuous_image(self):
        """
        reads a single image, convert it to Lab space and do not discretice its
        """
        # just keep trying until a valid image comes back
        while True:
            try:
                raw_img = imageio.imread(self._source_files[self.__counter])
                self.__counter = (self.__counter + 1) % len(self._source_files)
                tmp = self._random_rotation(raw_img)
                tmp = self._random_crop(tmp)

                tmp = np.clip(tmp.astype(np.float32), 0.0, 1.0)
                tmp = color.rgb2lab(tmp)  # also converts uint8 to float64
                tmp = tmp.astype(np.float32) / 100
                return tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2]
            except Exception as e:
                # print(e)
                pass

    def get_single_discrete_image(self):
        # discretice a and b channel
        ch_L, ch_a, ch_b = self.get_single_continuous_image()
        ch_a = encode_one_hot_a(ch_a)
        ch_b = encode_one_hot_b(ch_b)
        return ch_L, ch_a, ch_b

    def get_batch(self, continuous=True, only_L_ch=False, as_tuple=True):
        batch_L, batch_a, batch_b = [], [], []
        for k in range(self._batch_size):
            if continuous:
                ch_L, ch_a, ch_b = self.get_single_continuous_image()
            else:
                ch_L, ch_a, ch_b = self.get_single_discrete_image()
            batch_L.append(ch_L)
            batch_a.append(ch_a)
            batch_b.append(ch_b)

        if continuous:
            output_shape = self._batch_shape_1
        else:
            output_shape = (BATCH_SIZE, TARGET_SIZE, TARGET_SIZE, BINS)
        batch_L = np.stack(batch_L, 0).reshape(self._batch_shape_1)
        batch_a = np.stack(batch_a, 0).reshape(output_shape)
        batch_b = np.stack(batch_b, 0).reshape(output_shape)

        if only_L_ch:
            return batch_L
        else:
            if as_tuple:
                return batch_L, batch_a, batch_b
            else:
                return np.concatenate([batch_L, batch_a, batch_b], axis=-1)

    def _random_crop(self, image):
        """
        crops to a random subimage, expects three channels
        """
        rows, cols, ch = image.shape
        if not ch == 3:
            # print("wrong number of channels")
            raise

        short_dim = min([rows, cols])
        if not short_dim >= self._target_size:
            # print("image too small")
            raise

        new_size = random.randint(self._target_size, short_dim)
        offset_rows = random.randint(0, rows - new_size)
        offset_cols = random.randint(0, cols - new_size)
        cropped_image = image[offset_rows:offset_rows + new_size, offset_cols:offset_cols + new_size, :]
        scale = self._target_size / float(new_size)
        new_image = transform.rescale(cropped_image, scale, anti_aliasing=True, multichannel=True, mode='constant')
        return new_image

    def _random_rotation(self, img):
        """ rotates and flips image, works with any number of channels
        """
        # horizontal flip
        if 1 == random.randint(0, 1):
            img = np.flip(img, axis=1)

        return img


def start_full_image_generator(data_queue, definitions_package):
    init(definitions_package)
    data_gen = ImageDataGenerator(source_files, BATCH_SIZE, TARGET_SIZE)
    print("continuous=True, only_L_ch=False, as_tuple=False")
    while True:
        data_queue.put(data_gen.get_batch(continuous=True, only_L_ch=False, as_tuple=False))


# data queues
def populate_trainer_queues(full_images_queue):
    tasks = []
    tasks.append(Process(target=start_full_image_generator, args=(full_images_queue, imported_definitions,)))
    for item in tasks:
        item.start()
    return tasks

