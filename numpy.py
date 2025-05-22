import builtins
import math

def sum(iterable):
    return builtins.sum(iterable)

def log2(x):
    return math.log2(x)

def diff(seq):
    return [seq[i+1] - seq[i] for i in range(len(seq)-1)]
