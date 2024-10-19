from .model import XOR_PNN

DATA_DIR = "data/xor_pnn/train"
SAVE_DIR = "experiments/trained_models/xor_pnn"

xor_pnn = XOR_PNN()
xor_pnn.train(data_dir=DATA_DIR)
xor_pnn.save_model(SAVE_DIR)
