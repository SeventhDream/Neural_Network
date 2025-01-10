import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

image_data = 'train.csv'
data = pd.read_csv(os.path.join(__location__, image_data))



