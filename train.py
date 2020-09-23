""" Contains functions regarding training """
from Seq2Seq.utils.data import generate_nmt_batches
from Seq2Seq.utils.helpers import compute_accuracy

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
                           sample_probability=args.sample_probability)

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
                           sample_probability=1.0)

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