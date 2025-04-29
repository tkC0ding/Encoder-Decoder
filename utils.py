import unicodedata
import re
import torch

SOS_token = 0
EOS_token = 1

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
    return torch.tensor(idx, dtype=torch.float32).view(1,-1)

def pairToTensor(pair, input_lang, output_lang):
    input_idx = strToTensor(pair[0], input_lang)
    target_idx = strToTensor(pair[1], output_lang)

    return (input_idx, target_idx)

