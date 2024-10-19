from .model import PNN

DATA_DIR = "data/pnn/train"
SAVE_DIR = "experiments/trained_models/pnn"

pnn = PNN()
pnn.train(data_dir=DATA_DIR)
pnn.save_model(SAVE_DIR)
