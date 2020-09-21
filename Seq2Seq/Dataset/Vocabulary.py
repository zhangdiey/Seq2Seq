class Vocabulary(object):
    """ Class to process text and extract Vocabulary for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ Returns a dictionary that can be serialized """
        return {"token_to_idx": self._token_to_idx,
                "add_unk": self._add_unk,
                "unk_token": self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ Instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """ Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index
    
    def lookup_token(self, token):
        """ Retrieve the index associated with the token or the UNK
            index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >= 0 (having been added into the
              Vocabulary for the UNK functionality)
        """
        if self._add_unk:
            return self._token_to_idx.get(token)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """ Return the token associated with the inex

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str___(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

class SequenceVocabulary(Vocabulary):
    """ Class to process text and extract Vocabulary in Sequences for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>",
                 mask_token="<MASK>", sos_token="<SOS>", eos_token="<EOS>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
            mask_token (str): the MASK token to add into the Vocabulary
            sos_token (str): the SOS token to add into the Vocabulary
            eos_token (str): the EOS token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._sos_token = sos_token
        self._eos_token = eos_token

        self.mask_index = self.add_token(mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)
        self.sos_index = self.add_token(sos_token)
        self.eos_index = self.add_token(eos_token)

    def to_serializable(self):
        """ Returns a dictionary that can be serialized """
        contents = super().to_serializable()
        contents.update({'mask_token': self._mask_token,
                         'sos_token': self._sos_token,
                         'eos_token': self._eos_token})
        return contents

    def __str__(self):
        return "<SequenceVocabulary(size=%d)>" % len(self)