import torch
from model import Encoder, Decoder
from utils import data_loader, from_dict
from train_eval_utils import train
import pickle
from data_preprocessing import preprocess

PATH = "data/preprocessed.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
batch_size = 32

with open(PATH, 'rb') as file:
    data = pickle.load(file)

input_lang = preprocess()
output_lang = preprocess()

input_lang_data = data['input_lang_data']
output_lang_data = data['output_lang_data']
pairs = data['pairs']

input_lang.from_dict(input_lang_data)
output_lang.from_dict(output_lang_data)

input_lang, output_lang, train_dataloader = data_loader(input_lang, output_lang, pairs, batch_size)

encoder = Encoder(input_lang.n_words, hidden_size).to(device)
decoder = Decoder(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)