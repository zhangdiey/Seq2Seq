""" The entry of the framework where work shall be done """

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import load_tatoeba_dataset
from train import make_epoch_train_state, epoch_train
from Seq2Seq.models.Encoder import GRUEncoder
from Seq2Seq.models.Decoder import GRUDecoder
from Seq2Seq.models.NMTModel import BasicNMTModel

parser = argparse.ArgumentParser(description='Preprocessing the raw data.')
parser.add_argument("-p", "--path", help="path to the csv file")
parser.add_argument("-c", "--cuda", help="use GPU", default=True)
parser.add_argument("--source_embedding_dim", help="size of the source embedding vectors", default=256)
parser.add_argument("--target_embedding_dim", help="size of the target embedding vectors", default=256)
parser.add_argument("--encoder_rnn_hidden_size", help="size of the encoder RNN hidden state vectors", default=256)
parser.add_argument("--decoder_rnn_hidden_size", help="size of the decoder RNN hidden state vectors", default=512)
parser.add_argument("--bidirectional", help="flag of bidirectional GRU for encoder", default=True)
parser.add_argument("--encoder_num_layers", help="size of encoder RNN layers", default=1)
parser.add_argument("--learning_rate", help="learning rate", default=0.001)
parser.add_argument("--num_epochs", help="number of epochs", default=10)
parser.add_argument("--batch_size", help="batch size", default=64)
parser.add_argument("--sample_probability", help="probability of using sampling instead of teachers in decoder", default=0.5)
args = parser.parse_args()
print(args)

if args.path:
    path_to_file = args.path
else:
    raise Exception("The path to the csv file not privided!")

train_state = make_epoch_train_state(args)

# Set device
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# Dataset and vectorizer
dataset = load_tatoeba_dataset(path_to_file)
vectorizer = dataset.get_vectorizer()

# Print size of each split in the dataset
for key in dataset._lookup_dict:
    print(key, dataset._lookup_dict[key][1])

# Model
encoder = GRUEncoder(num_embeddings=len(vectorizer.source_vocab), 
                     embedding_dim=args.source_embedding_dim,
                     rnn_hidden_size=args.encoder_rnn_hidden_size,
                     bidirectional=args.bidirectional,
                     num_layers=args.encoder_num_layers)

decoder = GRUDecoder(num_embeddings=len(vectorizer.target_vocab), 
                     embedding_dim=args.target_embedding_dim,
                     rnn_hidden_size=args.decoder_rnn_hidden_size,
                     sos_index=vectorizer.target_vocab.sos_index)

basic_nmt = BasicNMTModel(encoder, decoder)
basic_nmt = basic_nmt.to(args.device)

# Loss and optimizer
ignore_index = vectorizer.target_vocab.mask_index
loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)
optimizer = optim.Adam(basic_nmt.parameters(), lr=args.learning_rate)

epoch_train(dataset=dataset, model=basic_nmt, loss_func=loss_func, 
            optimizer=optimizer, train_state=train_state, args=args)