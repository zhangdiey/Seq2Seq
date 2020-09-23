import argparse
import pandas as pd
import numpy as np
import collections
from Seq2Seq.Dataset.Dataset import Seq2SeqDataset
from Seq2Seq.Dataset.Vectorizer import Seq2SeqVectorizer
from Seq2Seq.Dataset.Vocabulary import SequenceVocabulary
from Seq2Seq.utils.data import preprocess_text

def load_tatoeba_dataset(csv, train_proportion=0.8, val_proportion=0.1,
                         test_proportion=0.1, random_seed=0):
    """ Load a tatoeba bilingual text into a dataset
        The format of the tatoeba data should be:
        #   Source  Target  Attribute   ...
        1   s1      t1      a1          ...
        2   s2      t2      a2          ...
        ...
        Each column should be seperated by a tab.
    
    Args:
        csv (str): path to the csv file
        train_proportion (0, 1): training set proportion (default: 0.8)
        val_proportion (0, 1): validation set proportion (default: 0.1)
        test_proportion (0, 1): test set proportion (default: 0.1)
        random_seed (int): random seed (default: 0)
    Returns:
        dataset (Seq2SeqDataset): the preprocessed dataset
    """
    raw_data = pd.read_csv(csv, names=["source_language", "target_language","attribute"],
                        usecols=["source_language", "target_language"] ,sep="\t", header=None)
    
    shuffled_data = raw_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    shuffled_data["split"] = ""

    n_total = len(shuffled_data)
    n_train = int(train_proportion * n_total)
    n_val = int(val_proportion * n_total)
    n_test = int(test_proportion * n_total)

    for index,row in shuffled_data.iterrows():
        if index < n_train:
            row["split"] = "train"
        elif index < n_train + n_val:
            row["split"] = "val"
        elif index < n_train + n_val + n_test:
            row["split"] = "test"
        else:
            # may happen due to rounding
            row["split"] = "test"

    shuffled_data["source_language"] = shuffled_data["source_language"].apply(preprocess_text)
    shuffled_data["target_language"] = shuffled_data["target_language"].apply(preprocess_text)

    vectorizer = Seq2SeqVectorizer.from_dataframe(shuffled_data)
    dataset = Seq2SeqDataset(shuffled_data, vectorizer)

    return dataset