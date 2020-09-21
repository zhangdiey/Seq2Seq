from torch.utils.data import Dataset
import pandas as pd
from .Vectorizer import Seq2SeqVectorizer

class Seq2SeqDataset(Dataset):
    def __init__(self, df, vectorizer):
        """
        Args:
            df (pandas.DataFrame): the dataset
            vectorizer (Seq2SeqVectorizer): vectorizer instantiated from dataset
        """

        self.df = df
        self._vectorizer = vectorizer

        self.train_df = self.df[self.df.split=="train"]
        self.train_size = len(self.train_df)

        self.val_df = self.df[self.df.split=="val"]
        self.val_size = len(self.val_df)
        
        self.test_df = self.df[self.df.split=="test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {"train": (self.train_df, self.train_size),
                             "val": (self.val_df, self.val_size),
                             "test": (self.test_df, self.test_size)}

        # default split
        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv):
        """ Load dataset and make a new vectorizer from scratch
        
        Args:
            csv (str): location of the dataset
        Returns:
            an instance of Seq2SeqDataset
        """
        df = pd.read_csv(csv)
        return cls(df, Seq2SeqVectorizer.from_dataframe(df))

    def get_vectorizer(self):
        """ Returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ Selects the splits in the dataset
        
        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dict of the data point's source language and target language
        """
        row = self._target_df.iloc[index]

        return self._vectorizer.vectorize(row["source_language"], row["target_language"])

    def get_num_batches(self, batch):
        """ Returns the number of batches in the dataset """
        return len(self) // batch