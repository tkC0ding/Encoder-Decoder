import torch
from model import Encoder, Decoder
from utils import data_loader
from train_eval_utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = data_loader(batch_size)

encoder = Encoder(input_lang.n_words, hidden_size).to(device)
decoder = Decoder(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)