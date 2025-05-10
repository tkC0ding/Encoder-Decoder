import torch
from torch import nn
import time
from utils import timeSince, showPlot
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    print(len(dataloader))
    for data in dataloader:
        input_tensor, output_tensor = data
        input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attention_weights = decoder(encoder_outputs, encoder_hidden, output_tensor)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), output_tensor.view(-1))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()

    return total_loss/len(dataloader)

def val_epoch(dataloader, encoder, decoder, criterion):
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            input_tensor, output_tensor = data
            input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, attention_weights = decoder(encoder_outputs, encoder_hidden, output_tensor)

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), output_tensor.view(-1))

            total_loss += loss.item()

    return total_loss/len(dataloader)


def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()

    plot_losses = []
    plot_losses_val = []

    print_loss_total = 0  
    plot_loss_total = 0  

    print_loss_total_val = 0  
    plot_loss_total_val = 0  

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        val_loss = val_epoch(val_dataloader, encoder, decoder, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        print_loss_total_val += val_loss
        plot_loss_total_val += val_loss


        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print_loss_avg_val = print_loss_total_val / print_every
            print_loss_total_val = 0

            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg, print_loss_avg_val))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every

            plot_loss_avg_val = plot_loss_total_val / plot_every
            
            plot_losses_val.append(plot_loss_avg_val)
            plot_losses.append(plot_loss_avg)

            plot_loss_total_val = 0
            plot_loss_total = 0

    showPlot(plot_losses)
    showPlot(plot_losses_val)