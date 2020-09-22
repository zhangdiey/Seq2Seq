""" This script split the data to create new train,
    val, and test splits. The format of the dataset 
    should be:
    #   Source  Target  Attribute   ...
    1   s1      t1      a1          ...
    2   s2      t2      a2          ...
    ...
    Each column should be seperated by a tab.
"""
import argparse
import pandas as pd
import numpy as np
import collections
from Seq2Seq.Dataset.Dataset import Seq2SeqDataset
from Seq2Seq.Dataset.Vectorizer import Seq2SeqVectorizer
from Seq2Seq.Dataset.Vocabulary import SequenceVocabulary

if __name__ == "__main__":
    """ For testing purposes only """
    my_parser = argparse.ArgumentParser(description='')

    RANDOM_SEED = 0

    path_to_file = "Data/Tatoeba/fra-eng/fra.txt"

    raw_data = pd.read_csv(path_to_file, names=["source_language", "target_language","attribute"],
                        usecols=["source_language", "target_language"] ,sep="\t", header=None)

    shuffled_data = raw_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    shuffled_data["split"] = ""

    n_total = len(shuffled_data)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = int(0.1 * n_total)

    for index,row in shuffled_data.iterrows():
        if index < n_train:
            row["split"] = "train"
        elif index < n_train + n_val:
            row["split"] = "val"
        elif index < n_train + n_val + n_test:
            row["split"] = "test"
        else:
            row["split"] = "test"

    vectorizer = Seq2SeqVectorizer.from_dataframe(shuffled_data)
    dataset = Seq2SeqDataset(shuffled_data, vectorizer)

    # Print the first item in training, validation and test dataset
    dataset.set_split("train")
    print("train: ", dataset.__getitem__(0))
    dataset.set_split("val")
    print("val: ", dataset.__getitem__(0))
    dataset.set_split("test")
    print("test: ", dataset.__getitem__(0))
