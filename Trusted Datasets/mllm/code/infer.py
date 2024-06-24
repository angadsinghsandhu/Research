## IMPORTS
import os
import torch
import deepspeed

# models
from code.nets.decoder_full import ESMDecoderFull
from code.nets.decoder_half import ESMDecoderHalf
from code.nets.mlp_decoder import ESMDecoderMLP
from code.nets.moe_decoder_full import ESMDecoderFullMOE
from code.nets.moe_decoder_half import ESMDecoderHalfMOE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the pre-trained ESM encoder
def load_esm_encoder(model_name="facebookresearch/esm", model_type="esm2_t33_650M_UR50D", out_layer=31):
    encoder, alphabet = torch.hub.load(model_name, model_type)
    batch_converter = alphabet.get_batch_converter()
    return encoder, batch_converter, out_layer

# Function to load the trained decoder model
def load_deepspeed_model(checkpoint_dir, checkpoint_tag, num_layers, embedding_dim, vocab_size, device, decoder_input, decoder_type, num_heads, batch_size):
    if decoder_input == "full":
        if decoder_type == "Full": model = ESMDecoderFull(vocab_size=vocab_size, num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)
        elif decoder_type == "FullMoE": model = ESMDecoderFullMOE(vocab_size=vocab_size, num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)
    elif decoder_input == "half":
        if decoder_type == "MLP": model = ESMDecoderMLP(embed_size=embedding_dim, num_layers=num_layers*2, hidden_size=batch_size).to(device)
        elif decoder_type == "Half": model = ESMDecoderHalf(embed_size=embedding_dim, num_layers=num_layers, num_heads=num_heads).to(device)
        elif decoder_type == "HalfMoE": model = ESMDecoderHalfMOE(num_layers=num_layers, embed_size=embedding_dim, num_heads=num_heads).to(device)
    
    # Initialize DeepSpeed engine
    config = {
        "train_batch_size": 256,
        "gradient_accumulation_steps": 1,
    }

    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=model.parameters(),
        config_params=config
    )
    
    # Load the DeepSpeed checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_tag)
    # It typically expects the checkpoint folder, not a file
    _, client_state = model_engine.load_checkpoint(checkpoint_dir, tag=checkpoint_tag, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
    
    return model_engine

# Function to perform inference
def inference(encoder, decoder, batch_converter, sequences, device="cuda", out_layer=30, decoder_input='half'):
    # Prepare input data using batch_converter
    conv = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
    _, _, inputs = batch_converter(conv)
    inputs = inputs.to(decoder.device)

    # Encode the input sequences
    with torch.no_grad():
        esm_out = encoder(inputs, repr_layers=[out_layer], return_contacts=True)
        emb = esm_out["representations"][out_layer]

    # Make predictions using the decoder model
    decoder.eval()
    with torch.no_grad():
        if decoder_input == 'half':
            predictions = decoder(emb)
        elif decoder_input == 'full':
            predictions = decoder(inputs, emb)

    return predictions.cpu().numpy()

if __name__ == "__main__":

    sequences = [
        "KRKRTRFTPEQLAVLEAYFAKNPYPSKEEREELAKELGLTEKQVKVWFQNRRAKERR",
        "RRKRTRFTPEQLEILERLFAKNPYPSREEREELAEELGLSERQVKVWFQNRRAKEKR",
        "KRKRTRFTPEQLEILEAEFQKNPYPSREEREELAKELGLSERQVQVWFQNRRAREKR",
        "RRKRTRFTPEQLEVLEKAFQENPYPSREEIEELAKELGLSERQVKVWFQNRRKKERK",
        "KRKRTRFTPEQLEILEAIFKQNPYPSREEREELAKELGLSEKQVKVWFQNRRAKERK",
        "RRKRTRFTPEQLEILEAAFAKNPYPSREEREELAKELGLSERQVKVWFQNRRAKEKR",
        "RRKRTTFTPEQLEELEKEFEENPYPDRERREELARRLGLTERQVQVWFQNRRAKWKK",
        "RRKRTTFTPEQLEELEKAFQRTHYPDVFTREELAARLGLTERRVQVWFQNRRAKWRK",
        "RRSRTTFTPEQLEELEKAFEKTHYPDVFEREELAARLGLTEARVQVWFQNRRAKWRK",
        "RRSRTTFTPEQLEELEKAFERTHYPDVFAREELAARLGLTEARVQVWFQNRRAKWRK",
        "RRKRTTFTPEQLEELEKAFEKNHYPDVEEREELAKKLGLTERQVQVWFQNRRAKWKK",
        "KKPRTFYSADQLEELEKMFQEDHYPDNEKRREIAAAVGVTPQRILVWFQNRRAKWRK",
        "KRHRTRFTPAQLNELERSFAKTHYPDIFMREELALRIGLTESRVQVWFQNRRAKWKK",
        "KKPRHRHSPAQLAALNELFEKDEHPALELRQSLAERLGMETKTVNAWFQNKRASSKK"
    ]

    labels = [
        13.4,
        8.40,
        11.5,
        14.5, 
        9.74,
        12.2,
        10.5,
        8.96,
        8.65,
        8.19,
        11.1,
        5.32,
        5.45,
        2.97
    ]


    checkpoint_dir = "./checkpoints"
    checkpoint_tag = "ESMnrgFull_e1_t2-esm2_t33_650M_UR50D"
    model_name = "facebookresearch/esm"
    model_type = "esm2_t33_650M_UR50D"
    out_layer = 31
    num_layers = 2
    decoder_type = "Full"        # 'Full', 'Half', 'MLP', 'FullMoE', 'HalfMoE'
    decoder_input = "full"          # 'full', 'half'
    num_heads = 64
    batch_size = 256
    embedding_dim = 1280
    vocab_size = 33

    # Load the ESM encoder and the trained decoder model
    encoder, batch_converter, out_layer = load_esm_encoder(model_name=model_name, model_type=model_type, out_layer=out_layer)
    
    decoder = load_deepspeed_model(
        checkpoint_dir=checkpoint_dir, checkpoint_tag=checkpoint_tag, 
        num_layers=num_layers, embedding_dim=embedding_dim, vocab_size=vocab_size, 
        device=device, decoder_input=decoder_input, decoder_type=decoder_type, 
        num_heads=num_heads, batch_size=batch_size)

    # Perform inference
    predictions = inference(encoder.to(decoder.device), decoder, batch_converter, sequences, device=device, out_layer=out_layer, decoder_input=decoder_input)

    # Print the results
    for i, seq in enumerate(sequences):
        print(f"Sequence: {seq}")
        print(f"Label: {labels[i]}")
        print(f"Prediction: {predictions[i][0]:.4f}\n")