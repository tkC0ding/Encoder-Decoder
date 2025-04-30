import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 4, batch_first=True)
        nn.ModuleList([self.embedding, self.gru])
    
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.maxlength = 100
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size, 4, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.RelU = nn.ReLU()
        self.layer_1 = nn.Linear(2*hidden_size, 3)
        self.layer_2 = nn.Linear(3, 1)
        self.softmax = nn.Softmax(dim=0)

        nn.ModuleList([self.embedding, self.gru, self.out, self.RelU, self.layer_1, self.layer_2, self.softmax])

    def single_step(self, input, hidden, encoder_hidden):

        a = torch.zeros_like(encoder_hidden).to(device)
        a[:] = hidden[0][-1]
        enc = torch.cat([a, encoder_hidden], dim=1)

        output = self.layer_1(enc)
        output = self.layer_2(output)
        allignment_scores = self.softmax(output)

        C = (encoder_hidden * allignment_scores).sum(dim=0)

        embeddings = self.embedding(input)
        embeddings = self.RelU(embeddings)

        fake = torch.zeros_like(embeddings).to(device)
        fake[:, :] = C
        embeddings = torch.cat([embeddings, fake], dim=-1)

        output, hidden = self.gru(embeddings, hidden)
        output = self.out(output)
        return(output, hidden, allignment_scores)
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        encoder_outputs, encoder_hidden = encoder_outputs.to(device), encoder_hidden.to(device)
        batch_size = encoder_outputs.shape[0]
        updated_encodder_hidden = encoder_hidden[0]
        decoder_input = torch.empty((batch_size, 1), dtype=torch.int32).fill_(0).to(device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.maxlength):
            decoder_output, decoder_hidden, allignment_scores = self.single_step(decoder_input, decoder_hidden, updated_encodder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1).to(device)
            else:
                _, index = decoder_output.topk(1)
                decoder_input = index.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, -1)
        return(decoder_outputs, decoder_hidden, allignment_scores)