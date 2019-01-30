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

ALPHA = 0.1  # alpha for leay relu

BINS = 16 # number of bins for one-hot-encoding a/b channels (needed in direct training)

LAMBDA_GP = 1E3  # weight for gradient penalty
LEARNING_RATE = 1E-6 # learning rate, used for all compiled models and Adam optimizer

# control for number of critic iterations
CRITIC_ITERATIONS_MIN = 5  # minimum number of critic training iterations, paper suggests 5
CRITIC_ITERATIONS_MAX = 32	# maximum number
CRITIC_ITERATIONS_INIT = 16
GENERATOR_ITERATIONS = 2 # number of generator iterations, paper suggets 1

THRESHOLD_A = 0.5  # threshold for critic performance to allow one training step on generator
THRESHOLD_B = 0.7  # threshold for critic performance to reduce training ratio

# pointers to training data
root_dir = "D:\\downloads\\places365\\2016\\f\\field"
source_folders = ['cultivated', 'wild', 'animals']
