
# training
LEARNING_RATE = 3e-4 
EPOCHS = 30
BATCH_SIZE = 128
EPOCH_BATCHES = None

# model
NEIGHBOR_RADIUS = 10000
OB_HORIZON = 8
PRED_HORIZON = 12
# group name of inclusive agents; leave empty for all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = ["PLAYER"]

# evaluation
WORLD_SCALE = 0.3048 # foot to meter