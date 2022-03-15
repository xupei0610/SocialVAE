
# training
LEARNING_RATE = 3e-4 
EPOCHS = 100    # it is better to use 30 fo univ to prevent overfitting
BATCH_SIZE = 128
EPOCH_BATCHES = 500 # number of batches per epoch, None for data_length//batch_size

# model
NEIGHBOR_RADIUS = 2
OB_HORIZON = 8
PRED_HORIZON = 12
# ids of ignored agents
EXCLUSIVE_AGENTS = []

# evaluation
WORLD_SCALE = 1