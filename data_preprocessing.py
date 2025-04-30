import pickle
import os
from utils import readData

DATAFOLDER = 'data'
FILENAME = 'eng-fra.txt'

class preprocess:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0:"SOS", 1:"EOS", 2:"UNK"}
        self.wordCount = {}
        self.n_words = 3

    def addSentence(self, sentence:str):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.wordCount[word] = 1
            self.n_words += 1
        else:
            self.wordCount[word] += 1
    
    def to_dict(self):
        return {
            'word2index' : self.word2index,
            'index2word' : self.index2word,
            'wordCount' : self.wordCount,
            'n_words' : self.n_words
        }
    
    @classmethod
    def from_dict(obj, data:dict):
        obj.word2index = data['word2index']
        obj.index2word = data['index2word']
        obj.wordCount = data['wordCount']
        obj.n_words = data['n_words']

def main(filepath):
    pairs = readData(filepath)
    input_lang = preprocess()
    output_lang = preprocess()
    for i in pairs:
        input_lang.addSentence(i[0])
        output_lang.addSentence(i[1])
    
    input_lang_data = input_lang.to_dict()
    output_lang_data = output_lang.to_dict()

    data = {"input_lang_data" : input_lang_data, "output_lang_data" : output_lang_data, "pairs" : pairs}

    with open("data/preprocessed.pkl", 'wb') as file:
        pickle.dump(data, file)
    
    print(f"Data has been preprocessed and saved in {DATAFOLDER} folder")


main(os.path.join(DATAFOLDER, FILENAME))