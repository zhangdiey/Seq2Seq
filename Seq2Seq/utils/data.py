from torch.utils.data import DataLoader

def generate_batches(dataset, batch_size, shuffle=True,
                         drop_last=True, device="cpu"):
    """ A generator function which wraps the PyTorch DataLoader """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def generate_nmt_batches(dataset, batch_size, shuffle=True,
                         drop_last=True, device="cpu"):
    """ A generator function which wraps the PyTorch DataLoader; NMT version """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        lengths = data_dict["source_length"].numpy()
        sorted_length_indices = lengths.argsort()[:-1].tolist()
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict
