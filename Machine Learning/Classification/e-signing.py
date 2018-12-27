""" predicting the likelihood of e-signing a loan based on financial history """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')

