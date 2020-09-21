from .Vocabulary import SequenceVocabulary
import numpy as np

class Seq2SeqVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use """

    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        Args:
            source_vocab (SequenceVocabulary): maps source words to integers
            target_vocab (SequenceVocabulary): maps target words to integers
            max_source_length (int): the longest sequence in the source dataset
            max_target_length (int): the longest sequence in the target dataset
        """

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length


    @classmethod
    def from_dataframe(cls, bitext_df):
        """ Instantiate the vectorizer from the dataset dataframe

        Args:
            bitext_df (pandas.DataFrame): the parallel text dataset
        Returns:
            an instance of the NMTVectorizer
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        max_source_length, max_target_length = 0, 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                source_vocab.add_token(token)
        
        return cls(source_vocab, target_vocab, max_source_length, max_target_length)
        

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """ Vectorize the provided indices

        Args:
            indices (list): a list of integers that represent a sequence
            vector_length (int): force the length of the index vector
            mask_index (int): the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        
        return vector

    def _get_source_indices(self, text):
        """ Return the vectorized source text

        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            indices (list): list of integers representing the text
        """
        indices = [self.source_vocab.sos_index]
        indices.extend(self.source_vocab.lookup_token(token)
                       for token in text.split(" "))
        indices.append(self.source_vocab.eos_index)

        return indices

    def _get_target_indices(self, text):
        """ Return the vectorized target text

        Args:
            text (str): the target text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list): list of integers; observations in target decoder
                y_indices (list): list of integers; predictions in target decoder
        """
        indices = [self.target_vocab.lookup_token(token)
                   for token in text.split(" ")]
        x_indices = [self.target_vocab.sos_index] + indices
        y_indices = indices + [self.target_vocab.eos_index]

        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """ Return the vectorized source and target text

        Args:
            source_text (str): text from the source language
            target_text (str): text from the target language
            use_dataset_max_lengths (bool): whether to use the max vector lengths

        Returns:
            The vectorized data point as a dictionary with the keys:
                source_vector, target_x_vector, target_y_vector, source_length
        """
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)

        return {"source_vector": source_vector,
                "target_x_vector": target_x_vector,
                "target_y_vector": target_y_vector,
                "source_length": len(source_indices)}

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])
        max_source_length = contents["max_source_length"]
        max_target_length = contents["max_target_length"]

        return cls(source_vocab=source_vocab, target_vocab=target_vocab, 
                   max_source_length=max_source_length, max_target_length=max_target_length)

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(),
                "target_vocab": self.target_vocab.to_serializable(),
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}