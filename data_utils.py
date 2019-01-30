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

"""
various weights for training the network via cross-entropy loss
determined by weights.script
"""

"""
# empty: use these weights if classes are not to be rebalanced
def magic_weight_a(k):
    return 0

def magic_weight_b(k):
    return 0
"""

"""
### marsh    
def magic_weight_a(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 2.14079243e-02, 7.69449747e-04, 9.99641801e-05,
       1.69134331e-04, 4.54734177e-03, 3.29466126e-02, 5.81816337e-02,
       1.10502086e-01, 7.71375853e-01, 0.00000000e+00, 0.00000000e+00]
    return bias[int(k)]


def magic_weight_b(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.93511234e-01,
       1.74261291e-01, 3.50253244e-03, 3.93268867e-04, 7.55688999e-05,
       6.25989687e-05, 1.13438328e-04, 2.32178427e-04, 6.81042010e-04,
       1.91876286e-03, 2.52480847e-02, 0.00000000e+00, 0.00000000e+00]
    return bias[int(k)]
"""

## baseball
"""
def magic_weight_a(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.15632114e-01,
       2.09718504e-04, 3.25220747e-05, 6.64449449e-06, 4.06973463e-06,
       4.45381022e-06, 2.15736675e-05, 9.33794910e-05, 2.25414488e-04,
       4.21655768e-04, 1.46349336e-03, 5.81884960e-01, 0.00000000e+00]
    return bias[int(k)]


def magic_weight_b(k):
    bias = [0.00000000e+00, 0.00000000e+00, 1.01229331e-01, 1.02811039e-02,
       1.45750504e-04, 1.79057256e-04, 2.68935341e-05, 5.76787608e-06,
       5.47713929e-06, 5.85206994e-06, 6.52467513e-06, 1.86319126e-05,
       7.57072517e-05, 1.06990350e-02, 8.77320868e-01, 0.00000000e+00]
    return bias[int(k)]
"""

## fields

def magic_weight_a(k):
    bias = [0.00000000e+00, 0.00000000e+00, 3.32578938e-01, 6.10656759e-04,
       2.70747074e-04, 2.03407529e-05, 6.44507844e-06, 2.86005760e-06,
       5.65342429e-06, 3.58396959e-05, 7.39372378e-05, 2.37789928e-04,
       1.93641303e-04, 8.05275878e-04, 6.65157875e-01, 0.00000000e+00]
    return bias[int(k)]


def magic_weight_b(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.93575049e-02,
       3.04246025e-03, 5.17736531e-04, 2.42757739e-04, 9.13337240e-05,
       1.17637676e-04, 1.05039973e-04, 1.49733234e-04, 2.32228107e-04,
       7.23111819e-04, 1.26761677e-03, 2.38576176e-03, 9.11767078e-01]
    return bias[int(k)]


###mansion
"""
def magic_weight_a(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.11069994e-01, 2.55249152e-03, 1.14346681e-04,
       1.55870775e-04, 6.34352761e-03, 2.71267232e-02, 6.52637046e-01,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    return bias[int(k)]


def magic_weight_b(k):
    bias = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.63229537e-02,
       2.47880043e-04, 1.59476832e-04, 1.24231421e-04, 3.24795803e-05,
       2.94615284e-05, 9.71081423e-05, 5.37695955e-04, 6.57326874e-03,
       9.15875444e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    return bias[int(k)]
"""

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
    for k in range(BINS):
        res[:, :, k] = np.multiply(res[:, :, k], 0.05 + magic_weight_a(k))
    return res


def encode_one_hot_b(img):
    res = encode_one_hot(img)
    for k in range(BINS):
        res[:, :, k] = np.multiply(res[:, :, k], 0.05 + magic_weight_b(k))
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

        # random rotation
        angle = random.randint(-2, 2) * 5
        if not angle == 0:
            img = transform.rotate(img, angle)
        return img


# saving files
def save_continuous_images(data_source, model):
    print("saving continuous image")
    batch_Lab = data_source.get()
    for ind in range(min(4, BATCH_SIZE)):  # batch_Lab.shape[0]):
        L_orig = batch_Lab[ind, :, :, 0]
        a_orig = batch_Lab[ind, :, :, 1]
        b_orig = batch_Lab[ind, :, :, 2]

        Lab_orig = np.stack([L_orig, a_orig, b_orig], axis=2)
        rgb_orig = color.lab2rgb((Lab_orig * 100).astype(np.float64))

        result = model.predict(batch_Lab[ind, :, :, 0].reshape((1, TARGET_SIZE, TARGET_SIZE, 1)))
        Lab_gen = result[0, :, :, :]
        rgb_gen = color.lab2rgb((Lab_gen * 100).astype(np.float64))
        rgb_gen = np.clip(rgb_gen, 0, 1)
        rgb_orig = np.clip(rgb_orig, 0, 1)
        try:
            fig = plt.figure()
            plt.imshow(rgb_gen)
            fig.savefig("generated_%d.png" % ind)
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(rgb_orig)
            fig.savefig("orig_%d.png" % ind)
            plt.close(fig)
        except Exception as e:
            print("failed generating and saving image")
            print(e)


def save_discrete_images(data_source, model):
    print("saving discrete image")
    batch_L, batch_a, batch_b = data_source.get()
    pred_discrete_a, pred_discrete_b = model.predict(batch_L)
    for ind in range(min(4, BATCH_SIZE)):  # batch_L.shape[0]):
        L_orig = batch_L[ind, :, :, 0]
        a_orig = decode_bin(np.argmax(batch_a[ind, :, :, :], axis=-1))
        b_orig = decode_bin(np.argmax(batch_b[ind, :, :, :], axis=-1))

        Lab_orig = np.stack([L_orig, a_orig, b_orig], axis=2)
        rgb_orig = color.lab2rgb((Lab_orig * 100).astype(np.float64))

        a_gen = decode_bin(np.argmax(pred_discrete_a[ind, :, :, :], axis=-1))
        b_gen = decode_bin(np.argmax(pred_discrete_b[ind, :, :, :], axis=-1))
        Lab_gen = np.stack([L_orig, a_gen, b_gen], 2)
        rgb_gen = color.lab2rgb((Lab_gen * 100).astype(np.float64))
        rgb_gen = np.clip(rgb_gen, 0, 1)
        rgb_orig = np.clip(rgb_orig, 0, 1)
        try:
            fig = plt.figure()
            plt.imshow(rgb_gen)
            fig.savefig("generated_%d.png" % ind)
            plt.close(fig)
            fig = plt.figure()
            plt.imshow(rgb_orig)
            fig.savefig("orig_%d.png" % ind)
            plt.close(fig)
        except Exception as e:
            print("failed generating and saving image")
            print(e)


def start_full_image_generator(data_queue, definitions_package):
    init(definitions_package)
    data_gen = ImageDataGenerator(source_files, BATCH_SIZE, TARGET_SIZE)
    print("continuous=True, only_L_ch=False, as_tuple=False")
    while True:
        data_queue.put(data_gen.get_batch(continuous=True, only_L_ch=False, as_tuple=False))


def start_L_image_generator(data_queue, definitions_package):
    init(definitions_package)
    data_gen = ImageDataGenerator(source_files, BATCH_SIZE, TARGET_SIZE)
    print("continuous=True, only_L_ch=True")
    while True:
        data_queue.put(data_gen.get_batch(continuous=True, only_L_ch=True))


def start_discrete_image_generator(data_queue, definitions_package):
    init(definitions_package)
    data_gen = ImageDataGenerator(source_files, BATCH_SIZE, TARGET_SIZE)
    print("continuous=False, only_L_ch=False")
    while True:
        data_queue.put(data_gen.get_batch(continuous=False, only_L_ch=False))


# data queues
def populate_trainer_queues(full_images_queue, only_L_ch_queue):
    tasks = []
    tasks.append(Process(target=start_full_image_generator, args=(full_images_queue, imported_definitions,)))
    tasks.append(Process(target=start_full_image_generator, args=(full_images_queue, imported_definitions,)))
    tasks.append(Process(target=start_full_image_generator, args=(full_images_queue, imported_definitions,)))
    tasks.append(Process(target=start_L_image_generator, args=(only_L_ch_queue, imported_definitions,)))
    tasks.append(Process(target=start_L_image_generator, args=(only_L_ch_queue, imported_definitions,)))
    tasks.append(Process(target=start_L_image_generator, args=(only_L_ch_queue, imported_definitions,)))
    for item in tasks:
        item.start()
    return tasks


# data queues
def populate_pre_train_queues(discrete_images_queue):
    tasks = []
    tasks.append(Process(target=start_discrete_image_generator, args=(discrete_images_queue, imported_definitions,)))
    tasks.append(Process(target=start_discrete_image_generator, args=(discrete_images_queue, imported_definitions,)))
    for item in tasks:
        item.start()
    return tasks
