# import torch

def predict(inputs):
    if type(inputs) is not list:
        return True
    return [True] * len(inputs)