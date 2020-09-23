""" The entry of the framework where work shall be done """

import argparse
from preprocessing import load_tatoeba_dataset

parser = argparse.ArgumentParser(description='Preprocessing the raw data.')
parser.add_argument("-p", "--path", help="path to the csv file")
args = parser.parse_args()

if args.path:
    path_to_file = args.path
else:
    raise Exception("The path to the csv file not privided!")

dataset = load_tatoeba_dataset(path_to_file)

# Print size of each split in the dataset
for key in dataset._lookup_dict:
    print(key, dataset._lookup_dict[key][1])