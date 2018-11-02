# -*- coding: utf-8 -*-
import random

def train_test_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    
    test_size = options.pop('test_size', 0.25)
            
    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))
    
    split_point = int((len(arrays[0])+1)*test_size)
    if n_arrays == 1:
        dataset_x = arrays[0]
        x_train, x_test = dataset_x[split_point:], dataset_x[:split_point]
        return x_train, x_test
    if n_arrays == 2:
        dataset_x = arrays[0]
        dataset_y = arrays[1]
        x_train, x_test, y_train, y_test = dataset_x[:split_point], dataset_x[split_point:], dataset_y[:split_point], dataset_y[split_point:]
        return x_train, x_test, y_train, y_test