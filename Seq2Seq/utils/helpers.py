def describe(x):
    """ Print the details of a tensor """
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

def compute_accuracy(y_pred, y_true):
    """ Calculate accuracy between prediction and gold
    
    Args:
        y_pred (torch.Tensor): predictions (of a minibatch)
        y_true (torch.Tensor): gold (of a minibatch)
    Returns:
        acc (0, 1): accuracy
    """
    return 0.0