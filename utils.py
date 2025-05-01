import unicodedata
import re
import torch
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, Subset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import math

plt.switch_backend('agg')
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize(s:str):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readData(filepath:str):
    return([[normalize(j) for j in i] for i in [s.split('\t') for l in open(filepath) for s in l.strip('\n').split('\n')]])

def strToidx(s, lang):
    a = []
    for word in normalize(s).split(' '):
        if word not in lang.word2index:
            a.append(3)
        else:
            a.append(lang.word2index[word])
    return a

def strToTensor(s, lang):
    idx = strToidx(s, lang)
    idx.append(EOS_token)
    return torch.tensor(idx, dtype=torch.long).view(1,-1)

def pairToTensor(pair, input_lang, output_lang):
    input_idx = strToTensor(pair[0], input_lang).to(device)
    target_idx = strToTensor(pair[1], output_lang).to(device)

    return (input_idx, target_idx)

def data_loader(input_lang, output_lang, pairs, batch_size, train_ratio=0.4, val_ratio=0.2):
    n = len(pairs)

    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.float32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.float32)

    for i, (inp, tar) in enumerate(pairs):
        inp_i = strToidx(inp, input_lang)
        out_i = strToidx(tar, output_lang)

        inp_i.append(EOS_token)
        out_i.append(EOS_token)

        input_ids[i, :len(inp_i)] = inp_i
        target_ids[i, :len(out_i)] = out_i

    data = TensorDataset(torch.LongTensor(input_ids, device="cpu"), torch.LongTensor(target_ids, device="cpu"))

    total = len(data)
    indicies = torch.randperm(total)
    train_end = int(train_ratio*total)
    val_end = train_end + int(val_ratio * total)

    train_indices = indicies[:train_end]
    val_indices = indicies[train_end:val_end]
    test_indices = indicies[val_end:]

    train_dataset = Subset(data, train_indices)
    val_dataset = Subset(data, val_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return input_lang, output_lang, train_loader, val_loader, test_loader

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
