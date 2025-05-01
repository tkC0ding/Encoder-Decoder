import torch
from model import Encoder, Decoder
from utils import data_loader
from train_eval_utils import train
import pickle
from data_preprocessing import preprocess

PATH = "data/preprocessed.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
batch_size = 32
input_lang = preprocess()
output_lang = preprocess()

with open('data/preprocessed.pkl' , 'rb') as file:
    data = pickle.load(file)

input_data = data['input_data']
output_data = data['output_data']
pairs = data['pairs']

input_lang.from_dict(input_data)
output_lang.from_dict(output_data)

input_lang, output_lang, train_dataloader, val_loader, test_loader = data_loader(input_lang, output_lang, pairs, batch_size)

encoder = Encoder(input_lang.n_words, hidden_size).to(device)
decoder = Decoder(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, val_loader, encoder, decoder, 80, print_every=5, plot_every=5)