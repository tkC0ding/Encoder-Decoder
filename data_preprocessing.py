import unicodedata
import re
import pickle
import os

DATAFOLDER = 'data'
FILENAME = 'eng-fra.txt'

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


with open(os.path.join(DATAFOLDER, 'preprocessed.pkl'), 'wb') as file:
    pickle.dump(readData(os.path.join(DATAFOLDER, FILENAME)), file)

print(f'Data preprocessed, saved in {DATAFOLDER} folder as preprocessed.pkl')