import torch.nn as nn

class BasicNMTModel(nn.Module):
    """ A Basic Neural Machine Translation Model """
    def __init__(self, encoder, decoder):
        """ 
        Args:
            source_vocab_size (int): number of words in source language
            source_embedding_size (int): size of source embedding
            target_vocab_size (int): number of words in target language
            target_embedding_size (int): size of target embedding
            encoding_size (int): size of the encoder RNN
            target_sos_index (int): index for START-OF-SEQUENCE token
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability, 
                inference, beam_size):
        """ The forward pass of the model
        Args:
            x_source (torch.Tensor): the source text data tensor
                x_source.shape should be (batch, vectorizer.max_source_length)
            x_source_lengths (torch.Tensor): the length of the sequences in x_source
            target_sequence (torch.Tensor): the target text data tensor
            sample_probability (float): schedule sampling parameter
        Returns:
            decoded_states (torch.Tensor): prediction vectors at each output step
        """
        encoder_state, final_hidden_state = self.encoder(x_source, x_source_lengths)
        decoded_state = self.decoder(encoder_state=encoder_state,
                                     initial_hidden_state=final_hidden_state,
                                     target_sequence=target_sequence,
                                     sample_probability=sample_probability,
                                     inference=inference, 
                                     beam_size=beam_size)

        return decoded_state
    