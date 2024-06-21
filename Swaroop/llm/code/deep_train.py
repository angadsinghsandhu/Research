## IMPORTS
import os
import json
import wandb
import argparse
from tqdm import tqdm
import numpy as np

import torch
import deepspeed

from code.utils.dataloaders import get_esm_loader
from code.utils.metrics import calculate_mse_accuracy, EarlyStopping

# models
from code.nets.decoder_full import ESMDecoderFull
from code.nets.decoder_half import ESMDecoderHalf
from code.nets.mlp_decoder import ESMDecoderMLP
from code.nets.moe_decoder_full import ESMDecoderFullMOE
from code.nets.moe_decoder_half import ESMDecoderHalfMOE

## FILE DESCRIPTION
'''
This Python script is designed to train the ESMnrg model using a specific dataset. The training process includes
partial training of the encoder, leveraging a pretrained ESM model that is utilized for processing structural data.

The script sets up the necessary components such as the ESM encoder and the ESMnrg decoder, defines the training 
and validation loops, and provides functionality for testing the model, visualizing the results, and saving the 
final model state.

Usage:
    python -m code.train
    deepspeed --module code.train --deepspeed_config=code/config/ds_ESMnrg.json

Requirements:
    - torch: PyTorch library must be installed and properly configured.

This script assumes that all required data and model components are correctly set up and accessible in the 
script's environment.
'''

## DEFINE HYPERPARAMETERS
# Encoder hyperparamerters
encoder_name = "facebookresearch/esm"
# encoder_type = "esm2_t48_15B_UR50D"
encoder_type = "esm2_t33_650M_UR50D"
# encoder_type = "esm2_t6_8M_UR50D"
out_layer = 31

# Decoder hyperparamerters
EPOCHS = 1
learning_rate = 0.001
num_layers = 2
batch_size = 256
vocab_size = 33
embedding_dim = 1280
num_heads = 64
decoder_input = None
decoder_type = None
checkpoint_dir = "./checkpoints"
checkpoint_tag = None

MASTER_PORT = 26300
os.environ['MASTER_PORT'] = str(MASTER_PORT)

def main(args):
    global embedding_dim, vocab_size, batch_size
    global learning_rate, num_layers
    global decoder_input, decoder_type, checkpoint_tag

    decoder_type = args.decoder_type
    decoder_input = args.decoder_input

    checkpoint_tag = f"ESMnrg{decoder_type}_e{EPOCHS}_t{num_layers}-{encoder_type}"

    # get config
    with open(args.deepspeed_config, 'r') as file: data = json.load(file)

    # update hyperparamerters from config file
    learning_rate = data.get("lr", learning_rate)
    batch_size = data.get("train_batch_size", batch_size)

    # Set the GPU device according to local_rank
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    ## ADD MODELS
    encoder, alphabet = torch.hub.load(encoder_name, encoder_type)

    # update information from encoder
    embedding_dim = encoder.embed_dim
    vocab_size = encoder.alphabet_size
    batch_converter = alphabet.get_batch_converter()
    
    # decoder options
    assert embedding_dim % num_heads == 0
    # 'Full', 'Half', 'MLP', 'FullMoE', 'HalfMoE'
    if decoder_input == "full":
        if decoder_type == "Full": decoder = ESMDecoderFull(vocab_size=vocab_size, num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)
        elif decoder_type == "FullMoE": decoder = ESMDecoderFullMOE(vocab_size=vocab_size, num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)
    elif decoder_input == "half":
        if decoder_type == "MLP": decoder = ESMDecoderMLP(embed_size=embedding_dim, num_layers=num_layers*2, hidden_size=batch_size).to(device)
        elif decoder_type == "Half": decoder = ESMDecoderHalf(embed_size=embedding_dim, num_layers=num_layers, num_heads=num_heads).to(device)
        elif decoder_type == "HalfMoE": decoder = ESMDecoderHalfMOE(num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)

    ## GET DATA
    trainloader = get_esm_loader(converter=batch_converter, file_path="data/ds23_sm_resampled.csv", file_type="csv", batch_size=batch_size, shuffle=True, num_workers=4, dataset='train')
    valloader = get_esm_loader(converter=batch_converter, file_path="data/ds23_sm_resampled.csv", file_type="csv", batch_size=batch_size, shuffle=True, num_workers=1, dataset='val')
    testloader = get_esm_loader(converter=batch_converter, file_path="data/ds23_sm_resampled.csv", file_type="csv", batch_size=batch_size, shuffle=True, num_workers=1, dataset='test')

    # Print the length of the data loaders
    print("Data loaders created successfully...")
    print(f"len trainloader: {len(trainloader)}")
    print(f"len valloader: {len(valloader)}")
    print(f"len testloader: {len(testloader)}\n")

    ## GET MODEL, OPTIMIZER, TRAINLOADER, SCHEDULER

    # initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=decoder, model_parameters=decoder.parameters(), config=args.deepspeed_config)

    # send encoder to appropriate gpu
    encoder = encoder.to(model_engine.device)

    # loss function
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss() # Mean Absolute Error Loss
    huber_loss = torch.nn.SmoothL1Loss(beta=1.0) # Huber Loss
    criterion = mae_loss

    # Initialize wandb
    wandb.init(project="ESMnrg", config={"epochs": EPOCHS, "batch_size": batch_size, "learning_rate": learning_rate})
    wandb.watch(model_engine, log="all")  # Track gradients and weights

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    # Load checkpoint if it exists
    start_epoch = 0
    checkpoint_exists = os.path.exists(os.path.join(checkpoint_dir, checkpoint_tag))
    checkpoint_path, client_state = None, {}
    if checkpoint_exists:
        checkpoint_path, client_state = model_engine.load_checkpoint(checkpoint_dir, checkpoint_tag)
        start_epoch = client_state.get("epoch", 0)
        print(f"Checkpoint loaded: {checkpoint_tag}, starting at epoch {start_epoch}")
    else:
        print(f"No checkpoint found for {checkpoint_tag}, starting from scratch")

    early_stopper = EarlyStopping(patience=3, verbose=True)

    ## DEFINE LOOP FOR TRAINING AND VALIDATING
    for epoch in range(start_epoch, EPOCHS):  # loop over the dataset multiple times

        # training loop
        train_running_loss = 0.0
        train_accuracy = 0.0
        model_engine.train()  # Set the model to train mode

        # Wrap the training data iterator with tqdm for a progress bar
        train_loader_with_progress = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, data in train_loader_with_progress:
            # get the input data
            labels, seq_names, inputs = data
            inputs = inputs.to(model_engine.device)

            # Zero the gradients managed by DeepSpeed
            model_engine.zero_grad()

            # Encode the inputs using the ESM encoder
            with torch.no_grad():
                esm_out = encoder(inputs, repr_layers=[out_layer], return_contacts=True)
                emb = esm_out["representations"][out_layer]         # emb shape : [2, 74, 1280]

            # Forward pass
            if args.decoder_input == "half":
                outputs = model_engine(emb)
            elif args.decoder_input == "full":
                outputs = model_engine(inputs, emb)
            # shape of outputs: (batch_size, 1) -> [256, 1]

            labels = torch.tensor(list(map(lambda x: [float(x.split(':')[1])], labels))).to(model_engine.device)
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            model_engine.backward(loss)
            model_engine.step()

            # Accumulate loss
            train_running_loss += loss.item()

            # Calculate R-squared accuracy for the current batch
            _, batch_accuracy = calculate_mse_accuracy(outputs, labels)
            train_accuracy += batch_accuracy

            # print inputs and predictions every 100 batches
            if i%100==0:
                print("-------------")
                print("-- labels --")
                print(labels[:5])
                print("-- outputs --")
                print(outputs[:5])
                print("-------------")

            if i%(len(train_loader_with_progress)//5):
                # Step the scheduler when 20% of the epoch is completed
                lr_scheduler.step()
                wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})

            # update the progress bar
            train_loader_with_progress.set_postfix({"Train Loss": train_running_loss / (i + 1), "Train Accuracy": train_accuracy / (i + 1)})
 
            # Log training metrics to Weights & Biases
            wandb.log({"train_loss": train_running_loss / (i + 1), "train_accuracy": train_accuracy / (i + 1)})

        print(f'[{epoch + 1}] train loss: {train_running_loss / len(trainloader)}, train accuracy: {train_accuracy / len(trainloader)}')

        # Step the scheduler at the end of each epoch
        lr_scheduler.step()
        wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})

        # validation loop
        val_running_loss = 0.0
        val_accuracy = 0.0
        model_engine.eval()  # Set the model to eval mode

        # Wrap the validation data iterator with tqdm for a progress bar
        val_loader_with_progress = tqdm(enumerate(valloader), total=len(valloader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        with torch.no_grad():
            for i, data in val_loader_with_progress:
                # get the input data
                labels, seq_names, inputs = data
                inputs = inputs.to(model_engine.device)

                # Encode the inputs using the ESM encoder
                esm_out = encoder(inputs, repr_layers=[out_layer], return_contacts=True)
                emb = esm_out["representations"][out_layer]         # emb shape : [2, 74, 1280]

                # Forward pass
                if args.decoder_input == "half":
                    outputs = model_engine(emb)
                elif args.decoder_input == "full":
                    outputs = model_engine(inputs, emb)
                # shape of outputs: (batch_size, 1) -> [256, 1]

                # Forward pass
                labels = torch.tensor(list(map(lambda x: [float(x.split(':')[1])], labels))).to(model_engine.device)
                loss = criterion(outputs, labels)

                # Accumulate loss
                val_running_loss += loss.item()

                # Calculate R-squared accuracy for the current batch
                _, batch_accuracy = calculate_mse_accuracy(outputs, labels)
                val_accuracy += batch_accuracy

                # update the progress bar
                val_loader_with_progress.set_postfix({"Val Loss": val_running_loss / (i + 1), "Val Accuracy": val_accuracy / (i + 1)})

                # Log validation metrics to Weights & Biases
                wandb.log({"val_loss": val_running_loss / (i + 1), "val_accuracy": val_accuracy / (i + 1)})

        print(f'[{epoch + 1}] val loss: {val_running_loss / len(valloader)}, val accuracy: {val_accuracy / len(valloader)}')

        # Save checkpoint at the end of each epoch
        client_state = {"epoch": epoch + 1}
        model_engine.save_checkpoint(checkpoint_dir, checkpoint_tag, client_state)

        # Early stopping
        early_stopper(val_running_loss / len(valloader))
        if early_stopper.early_stop:
            print("Early stopping")
            break

    ## TEST LOOP
    test_running_loss = 0.0
    test_accuracy = 0.0
    model_engine.eval()  # Ensure the model is in eval mode

    # Wrap the test data iterator with tqdm for a progress bar
    test_loader_with_progress = tqdm(enumerate(testloader), total=len(testloader), desc="Testing")

    with torch.no_grad():
        for i, data in test_loader_with_progress:
            # Get the input data
            labels, seq_names, inputs = data
            inputs = inputs.to(model_engine.device)

            # Encode the inputs using the ESM encoder
            esm_out = encoder(inputs, repr_layers=[out_layer], return_contacts=True)
            emb = esm_out["representations"][out_layer]         # emb shape : [2, 74, 1280]

            # Forward pass
            if args.decoder_input == "half":
                outputs = model_engine(emb)
            elif args.decoder_input == "full":
                outputs = model_engine(inputs, emb)
            # shape of outputs: (batch_size, 1) -> [256, 1]

            # compare predictions and labels
            labels = torch.tensor(list(map(lambda x: [float(x.split(':')[1])], labels))).to(model_engine.device)
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            test_running_loss += loss.item()

            # Calculate R-squared accuracy for the current batch
            _, batch_accuracy = calculate_mse_accuracy(outputs, labels)
            test_accuracy += batch_accuracy

            # update the progress bar
            test_loader_with_progress.set_postfix({"Test Loss": test_running_loss / (i + 1), "Test Accuracy": test_accuracy / (i + 1)})

            # Log test metrics to Weights & Biases
            wandb.log({"test_loss": test_running_loss / (i + 1), "test_accuracy": test_accuracy / (i + 1)})

    print(f'Final test loss: {test_running_loss / len(testloader)}, test accuracy: {test_accuracy / len(testloader)}')

    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with DeepSpeed")
    parser.add_argument('--deepspeed_config', type=str, required=True, help="DeepSpeed configuration file path")
    parser.add_argument('--local_rank', type=int, help="Local rank passed by DeepSpeed")
    parser.add_argument('--decoder_input', choices=['half', 'full'], required=True, help="Decide weather to use Self or Cross Attention")
    parser.add_argument('--decoder_type', choices=['Full', 'Half', 'MLP', 'FullMoE', 'HalfMoE'], required=True, help="Decide which model to use")
    args = parser.parse_args()
    main(args)
