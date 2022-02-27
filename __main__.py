import pandas as pd
import numpy as np
import yaml

with open("config.yaml", "r") as stream:
    cl = yaml.safe_load(stream)

train = cl['train']
feature = cl['feature']

train_data=pd.read_csv(train,names=feature)
