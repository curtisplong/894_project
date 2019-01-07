import argparse

import pandas as pd

from lib.Dataset import Dataset 
from lib.OriginalModel import OriginalModel
from lib import utils

import keras

#############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Model Training", action="store_true")
#parser.add_argument("--test", "Model Testing")
args = parser.parse_args()

RANDOM_SEED=159
pd.set_option('display.max_colwidth', -1)

md = Dataset()
md.stats()

m = OriginalModel(data = md, name="originalmodel")
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()

