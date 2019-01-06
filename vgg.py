import argparse

import pandas as pd

from lib.Dataset import Dataset 
from lib.VggModel import VggModel
from lib import utils

import keras

#############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Model Training", action="store_true")
#parser.add_argument("--test", "Model Testing")
args = parser.parse_args()

RANDOM_SEED=159
pd.set_option('display.max_colwidth', -1)

md = Dataset(img_width=224, img_height=224)
md.stats()

m = VggModel(data = md, mlp = [16], batch = 8, dropout = 0.5, name="vgg16-16", optimizer='rmsprop', epochs=30)
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()
