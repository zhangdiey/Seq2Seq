""" Contains functions regarding training """

from Seq2Seq.utils.data import generate_nmt_batches
from Seq2Seq.utils.helpers import compute_accuracy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import load_tatoeba_dataset
from Seq2Seq.models.Encoder import GRUEncoder
from Seq2Seq.models.Decoder import GRUDecoder
from Seq2Seq.models.NMTModel import BasicNMTModel

def make_epoch_train_state(args):
    return {"epoch_index": 0,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": -1,
            "test_acc": -1}

def epoch_train(dataset, model, loss_func, optimizer, train_state, args):
    """ A basic epoch based training session """
    for epoch_index in range(args.num_epochs):
        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset

        # Setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split("train")
        batch_generator = generate_nmt_batches(dataset, batch_size=args.batch_size,
                                               device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # The training routine is 5 steps:

            # Step 1: Zero the gradients
            optimizer.zero_grad()

            # Step 2: Compute the output
            y_pred = model(x_source=batch_dict["source_vector"], 
                           x_source_lengths=batch_dict["source_length"], 
                           target_sequence=batch_dict["target_x_vector"], 
                           sample_probability=args.sample_probability,
                           inference=False,
                           beam_size=0)

            # Step 3: Compute the loss
            target_vocab_size = y_pred.shape[-1]
            y_pred = y_pred.reshape(-1, target_vocab_size)
            target_sequence = batch_dict["target_y_vector"]
            target_sequence = target_sequence.reshape(-1)
            loss = loss_func(y_pred, target_sequence)
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            # Step 4: Use loss to produce gradients
            loss.backward()

            # Step 5: Use optimizer to take gradient step
            optimizer.step()

            # Compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict["target_y_vector"])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        # Iterate over val dataset

        # Setup: batch generator, set loss and acc to 0, set eval mode on
        dataset.set_split("val")
        batch_generator = generate_nmt_batches(dataset, batch_size=args.batch_size,
                                               device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # The validation routine is 3 steps:

            # Step 1: Compute the output
            y_pred = model(x_source=batch_dict["source_vector"], 
                           x_source_lengths=batch_dict["source_length"], 
                           target_sequence=batch_dict["target_x_vector"], 
                           sample_probability=1.0,
                           inference=False,
                           beam_size=0)

            # Step 2: Compute the loss
            target_vocab_size = y_pred.shape[-1]
            y_pred = y_pred.reshape(-1, target_vocab_size)
            target_sequence = batch_dict["target_y_vector"]
            target_sequence = target_sequence.reshape(-1)
            loss = loss_func(y_pred, target_sequence)
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            # Step 3: Compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict["target_y_vector"])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        print("epoch", epoch_index)
        print("training loss", train_state["train_loss"][-1])
        print("val loss", train_state["val_loss"][-1])

        # save model
        torch.save(model, args.save_path+"{index}".format(index=epoch_index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the csv file")
    parser.add_argument("-c", "--cuda", help="use GPU", default=True)
    parser.add_argument("-s", "--save_path", help="path where the model is to be saved")
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
        raise Exception("The path to the csv file was not privided!")

    if not args.save_path:
        raise Exception("The path where the model is to be saved was not privided!")

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