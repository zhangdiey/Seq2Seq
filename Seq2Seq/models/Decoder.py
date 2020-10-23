import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def verbose_attention(query_vector, key_vectors, value_vectors):
    """
    Args:
        encoder_state_vectors: 3-dim tensor from encoder
        query_vector: hidden state in decoder
    """
    batch, num_vectors, vector_size = key_vectors.size()
    vector_scores = \
        torch.sum(key_vectors * query_vector.view(batch,
                                                  1,
                                                  vector_size),
                  dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = \
        value_vectors * vector_probabilities.view(batch,
                                                  num_vectors,
                                                  1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities

class BasicDecoder(nn.Module):
    """ Baisc encoder """

    def __init__(self):
        super().__init__()

    def _init_indices(self, batch):
        """ Return the START-OF-SEQUENCE index vector """
        return torch.ones(batch, dtype=torch.int64) * self.sos_index

    def _init_context_vectors(self, batch):
        """ Return a zeros vector for initializing the context """
        return torch.zeros(batch, self._rnn_hidden_size)

    def forward(self):
        pass

class GRUDecoder(BasicDecoder):
    """ GRU Decoder """

    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size, 
                 sos_index):
        """
        Args:
            num_embeddings (int): size of target vocabulary
            embedding_dim (int): size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors
            sos_index (int): START-OF-SEQUENCE index
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_dim,
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_dim + rnn_hidden_size, 
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.sos_index = sos_index
        self._sample_temperature = 3

    

    def forward(self, encoder_state, initial_hidden_state, target_sequence,
                sample_probability, inference, beam_size):
        """ The forward pass of the model
            
        Args:
            encoder_state (torch.Tensor): output of the Encoder
            initial_hidden_state (torch.Tensor): last hiddent state in the Encoder
            target_sequence (torch.Tensor): target text data tensor
            sample_probability (float): schedule sampling parameter
        Returns:
            output_vectors (torch.Tensor): prediction vectors at each output step
        """
        if target_sequence is None:
            sample_probability = 1.0
        else:
            # assume batch first
            # permute to (seq_len, batch)
            target_sequence = target_sequence.permute(1, 0)

        # use the last encoder state as the initial hiddent state
        h_t = self.hidden_map(initial_hidden_state)

        batch = encoder_state.size(0)
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch)
        # initialize first y_t word as SOS
        y_t_index = self._init_indices(batch)

        h_t = h_t.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)

        output_vectors = []
        # all cached tensors are moved from the GPU and stored for analysis
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.detach().cpu().numpy()

        output_sequence_size = target_sequence.size(0)
        for i in range(output_sequence_size):
            # a helper Boolean and the teacher y_t_index
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().data.numpy())

            # Step 3: Use current hidden vector to attend to encoder state
            context_vectors, p_attn = \
                verbose_attention(query_vector=h_t,
                                  key_vectors=encoder_state,
                                  value_vectors=encoder_state)

            # auxiliary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.detach().cpu().numpy())

            # Step 4: Use current hidden and context vectors to make a
            #         prediction for the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(prediction_vector)
            # Step 4-a: In inference use beam search
            if inference:
                pass
                        
            # Sampling for next loop
            p_y_t_index = F.softmax(score_for_y_t_index * self._sample_temperature, dim=1)
            y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            # auxiliary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors