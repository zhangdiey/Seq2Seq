""" Contains functions regarding inference """

from Seq2Seq.utils.data import generate_nmt_batches
from Seq2Seq.utils.helpers import compute_accuracy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import load_tatoeba_dataset
from Seq2Seq.models.NMTModel import BasicNMTModel

def infer(dataset, split, model, args):
    # Predict the selected split of the dataset
   
    if split is None:
        split = "test"

    dataset.set_split(split)
    batch_generator = generate_nmt_batches(dataset, batch_size=args.batch_size,
                                               device=args.device)

    model.eval()
    with open(args.output, "w+") as output:
        for batch_dict in batch_generator:
            y_pred = model(x_source=batch_dict["source_vector"], 
                           x_source_lengths=batch_dict["source_length"], 
                           target_sequence=batch_dict["target_x_vector"], 
                           sample_probability=1.0,
                           inference=True,
                           beam_size=args.beam_size)
            # output.write(y_pred)
            print(y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the csv file")
    parser.add_argument("-s", "--save_path", help="path to the model")
    parser.add_argument("-o", "--output", help="path to the output file")
    parser.add_argument("-c", "--cuda", help="use GPU", default=True)
    parser.add_argument("--beam_size", help="beam size", default=5)
    parser.add_argument("--batch_size", help="batch size", default=64)
    args = parser.parse_args()
    print(args)

    if args.path:
        path_to_file = args.path
    else:
        raise Exception("The path to the csv file was not privided!")

    if not args.save_path:
        raise Exception("The path to the model was not privided!")

    if not args.output:
        raise Exception("The path to the output file was not privided!")

    # Set device
    if not torch.cuda.is_available():
        args.cuda = False
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # load dataset and model
    dataset = load_tatoeba_dataset(path_to_file)
    model = torch.load(args.save_path)
    model = model.to(args.device)
    model.eval()

    infer(dataset=dataset, split="test", model=model, args=args)