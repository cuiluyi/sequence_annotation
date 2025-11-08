# model parameters
EMBED_SIZE = 128
HIDDEN_SIZE = 128
BATCH_SIZE = 1 # batch_size=1 for simplicity (avoid padding and truncating)

# training parameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# file paths
SAVE_DIR = "ckpts/"
TRAIN_DATA_FILE = "data/train.txt"
TEST_DATA_FILE = "data/test.txt"