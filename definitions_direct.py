# image dimensions
HEIGHT = 256
WIDTH = 256

BATCH_SIZE = 6

# several derived constants
TARGET_SIZE = HEIGHT
IMAGE_SHAPE_0 = (HEIGHT, WIDTH)
IMAGE_SHAPE_1 = (HEIGHT, WIDTH, 1)
IMAGE_SHAPE_2 = (HEIGHT, WIDTH, 2)
IMAGE_SHAPE_3 = (HEIGHT, WIDTH, 3)
BATCH_SHAPE_1 = (BATCH_SIZE, *IMAGE_SHAPE_1)
BATCH_SHAPE_2 = (BATCH_SIZE, *IMAGE_SHAPE_2)

RIM = 16 # the critic is ignoring the edges of the image
REDUCED_SHAPE_3 = (HEIGHT - 2 * RIM, WIDTH - 2 * RIM, 3)

BINS = 16 # number of bins for one-hot-encoding a/b channels

ALPHA = 0.1 # alpha for leay relu
LEARNING_RATE = 1E-6

# pointers to training data
root_dir = "D:\\downloads\\places365\\2016\\f\\field"
source_folders = ['wild', 'cultivated', 'animals']
