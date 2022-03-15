
# training
LEARNING_RATE = 3e-4 
EPOCHS = 150
BATCH_SIZE = 128
EPOCH_BATCHES = None

# model
NEIGHBOR_RADIUS = 10000
OB_HORIZON = 8
PRED_HORIZON = 12
# ids of ignored agents
# -1 for ball in nba dataset
EXCLUSIVE_AGENTS = [-1]

# evaluation
WORLD_SCALE = 0.3048 # foot to meter