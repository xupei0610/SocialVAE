# model
OB_RADIUS = 5       # observe radius, neighborhood radius
OB_HORIZON = 8      # number of observation frames
PRED_HORIZON = 12   # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
RNN_HIDDEN_DIM = 512

# training
LEARNING_RATE = 7e-4 
BATCH_SIZE = 128
EPOCHS = 600        # total number of epochs for training
EPOCH_BATCHES = 100 # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 400    # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20       # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate

# evaluation
# WORLD_SCALE = 1     # in unit of meters
# in unit of meters
WORLD_SCALE = []
import os, sys
data_dir = sys.argv[sys.argv.index("--test")+1]
data_dir = os.path.dirname(data_dir)
H = {}
H_file = os.path.join(data_dir, "H_SDD.txt")
with open(H_file, "r") as f:
    for row in f.readlines():
        item = row.split()
        if not item: continue
        if not "jpg" in item[0]: continue
        if not "A" in item[3]: continue
        scene = item[0][:-4]
        scale = float(item[-1])
        H[scene] = 1./scale
# if you are using a different testing split for SDD,
# the following needs to be modified.
for scene, n_trajs in [
    ("coupa_0", 323), 
    ("coupa_1", 235), 
    ("gates_2", 155), 
    ("hyang_0", 630), 
    ("hyang_1", 427), 
    ("hyang_3", 61), 
    ("hyang_8", 12), 
    ("little_0", 52), 
    ("little_1", 110), 
    ("little_2", 42), 
    ("little_3", 362), 
    ("nexus_5", 14), 
    ("nexus_6", 334), 
    ("quad_0", 10), 
    ("quad_1", 20), 
    ("quad_2", 30), 
    ("quad_3", 12)
]:
    scale = H[scene]
    WORLD_SCALE.extend([scale] * n_trajs)


