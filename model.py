import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.maxlength = 20

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 4, batch_first=True)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def hidden_init(self):
        return torch.zeros((4, self.maxlength, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.maxlength = 20
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 4, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.RelU = nn.ReLU()

    def single_step(self, input, hidden):
        embeddings = self.embedding(input)
        embeddings = self.RelU(embeddings)
        output, hidden = self.gru(embeddings, hidden)
        output = self.out(output)
        return(output, hidden)
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty((batch_size, 1), dtype=torch.int32).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.maxlength):
            decoder_output, decoder_hidden = self.single_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, index = decoder_output.topk(1)
                decoder_input = index.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, -1)
        return(decoder_outputs, decoder_hidden, None)