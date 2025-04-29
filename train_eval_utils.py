import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
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


