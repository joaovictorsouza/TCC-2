import random
import pickle
from itertools import chain
import numpy as np  # Substituímos torch por numpy
from glob import glob
from hparams import hp

print("Loading training files")

train_data = pickle.load(open(hp.pkl_train, 'rb'))
dev_data = pickle.load(open(hp.pkl_dev, 'rb'))

def pad(batch, maxlen):
    '''Pads to the longest sample'''
    return [sample + [0]*(maxlen-len(sample)) for sample in batch]

def get_batch(max_span, batch_size, n_classes, train=True):
    '''f
    Returns
    x: (N, T)
    y: (N,)
    '''
    contexts_li = train_data if train else dev_data

    x, y, maxlen = [], [], 0
    for _ in range(batch_size):
        label = random.randint(0, n_classes-1)  # randint: [a, b]
        try:
            contexts = contexts_li[label]  # list of lists of lists
        except IndexError:
            continue
        if len(contexts) == 0: continue
        ctx = random.choice(contexts)  # list of lists
        history_span = random.randint(1, len(ctx) + 1)
        history = ctx[-history_span:]  # lists

        history = list(chain.from_iterable(history))  # list
        history = history[-max_span+2:]  # [3, 4, 5, ...]
        history = [101] + history + [102]  # 101: [CLS], 102: [SEP]
        x.append(history)
        y.append(label)
        maxlen = max(maxlen, len(history))

    # Padding e conversão para NumPy arrays
    x = pad(x, maxlen)
    x = np.array(x, dtype=np.int32)  # Usamos NumPy em vez de torch
    y = np.array(y, dtype=np.int32)
    
    return x, y