import tensorflow as tf
import numpy as np
def encorde(text,depth = 0):
    """"Support batch or single str"""
    text = [dict[char.lower()] for char in text]
    length = [len(text)]

    if depth:
        return text, len(text)
    return text,length

def shuffle_labels(sequences,dtype=np.int32):
    pass