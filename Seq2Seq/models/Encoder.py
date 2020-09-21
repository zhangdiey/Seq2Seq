import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BasicEncoder(nn.Module):
    """ Baisc encoder """

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class GRUEncoder(BasicEncoder):
    """ GRU encoder """

    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size, bidirectional, num_layers=1):
        """
        Args:
            num_embeddings (int): size of source vocabulary
            embedding_dim (int): size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors
            bidirectional (bool): flag of bidirectional GRU
            num_layers (int): size of RNN layers
        """
        super().__init__()

        self.source_embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        self._bidirectional = bidirectional

        self._num_layers = num_layers

        self.rnn = nn.GRU(embedding_dim, rnn_hidden_size, bidirectional=bidirectional,
                          batch_first=True, num_layers=num_layers)

    def forward(self, x_source, x_lengths):
        """ The forward pass of the model
        
        Args:
            x_source (torch.Tensor): the input data tensor
                x_source.shape is (batch, seq_len)
            x_lengths (torch.Tensor): vector of the lengths for each item in batch
        Returns:
            a tuple: x_unpacked (torch.Tensor), x_rnn_h (torch.Tensor)
                x_unpacked.shape = (batch, seq_len, rnn_hidden_size * num_directions)
                x_rnn_h.shape = (batch, rnn_hidden_size * num_directions)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape = (num_items, )
        x_lengths = x_lengths.detach().cpu().numpy()
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        # x_rnn_out.shape = (seq_len, batch, num_directions * rnn_hidden_size)
        # x_rnn_h.shape = (num_layers * num_directions, batch, rnn_hidden_size)
        x_rnn_out, x_rnn_h = self.rnn(x_packed)
        # permute to (batch, num_layers * num_directions, rnn_hidden_size)
        x_rnn_h = x_rnn_h.permute(1, 0, 2)

        # flatten features; reshape to (batch, num_layers * num_directions * rnn_hidden_size)
        x_rnn_h = x_rnn_h.contiguous().view(x_rnn_h.size(0), -1)

        x_unpacked, _ = pad_packed_sequence(x_rnn_out, batch_first=True)

        return x_unpacked, x_rnn_h

