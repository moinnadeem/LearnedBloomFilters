from torch import nn
import string

class LearnedBloomFilter(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # X is of shape batch_size, seq_len, 1
        # X is also sorted of greatest length to least length.
        batch_size = x.shape(0)
        encoded = self.encoder(x)  # batch_size, seq_len, embedding_dim
        
        encoded = nn.utils.rnn.pad_packed_sequence(encoded, lengths, batch_first=True)
        output, _ = self.gru(encoded.view(1, batch_size, -1)) 
        output = self.decoder(output.view(batch_size, -1))
        return output
