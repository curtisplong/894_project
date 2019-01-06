import argparse

import pandas as pd

from lib.Dataset import Dataset 
from lib.ConvModel import ConvModel
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

m = ConvModel(data = md, convolutions = [16], pooling = [2], mlp = [8], batch = 8, dropout = 0.5, name="small-16", optimizer='rmsprop', epochs=50)
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()

m = ConvModel(data = md, convolutions = [32], pooling = [2], mlp = [8], batch = 8, dropout = 0.5, name="small-32", optimizer='rmsprop', epochs=50)
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()

m = ConvModel(data = md, convolutions = [64], pooling = [2], mlp = [8], batch = 8, dropout = 0.5, name="small-64", optimizer='rmsprop', epochs=50)
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()

m = ConvModel(data = md, convolutions = [128], pooling = [2], mlp = [8], batch = 8, dropout = 0.5, name="small-128", optimizer='rmsprop', epochs=50)
if args.train:
    m.train()
else:
    m.test()
keras.backend.clear_session()

#m = ConvModel(data = md, c = [128,64,32], m = [100,20], b = 8, name="conv128-20", epochs=30)
#if args.train:
#    m.train()
#else:
#    m.test()
#keras.backend.clear_session()

#m = ConvModel(data = md, c = [256,128,64], m = [100,20], b = 8, name="conv256-20", epochs=30)
#if args.train:
#    m.train()
#else:
#    m.test()
#keras.backend.clear_session()
