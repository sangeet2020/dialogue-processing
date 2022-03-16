import time
import json

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import OurModel
from utils.util import open_file, set_seed, epoch_time
from data.restaurant_dataset import Restaurant8kDataset


def train_epoch(model, criterion, optimizer, train_set, batch_size, use_cuda):
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    epoch_loss = 0
    for batch in tqdm(data_loader):
        # Initialize a batch
        encoded_slots = batch[0].cuda() if use_cuda else batch[0]
        slots_attn_masks = batch[1].cuda() if use_cuda else batch[1]
        encoded_utts = batch[2].cuda() if use_cuda else batch[2]
        utts_attn_masks = batch[3].cuda() if use_cuda else batch[3]
        true_labels = batch[4].cuda() if use_cuda else batch[4]
        true_labels = true_labels.squeeze(dim=1) # [32, 50]
        
        # Training step
        output = model(encoded_slots, slots_attn_masks, encoded_utts, utts_attn_masks)

        loss = criterion(output, true_labels)
        epoch_loss += loss.item()

        optimizer.zero_grad(); 
        loss.backward(); 
        optimizer.step()
        
    return epoch_loss / len(data_loader)


def train(config):
    set_seed(config["seed"])
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    model_name = config["model_name"]

    # Initialize dataset.
    train_set = Restaurant8kDataset(config["train_path"],
                                    config["bert_model"],
                                    config["slot_max_len"],
                                    config["utt_max_len"],
                                    config["include_all"])
    print(f"Number of training examples: {len(train_set)}")

    device = torch.device('cuda' if config["use_cuda"] else 'cpu')

    # Model initialization
    model = OurModel(config, train_set.num_of_bio_tags(), device).to(device)

    # Optimizer and loss function initialization
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.get_tag_id('P'))
    
    # Training loop
    train_losses = []
    best_train_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()

        # Epoch training step
        train_loss = train_epoch(model, criterion, optimizer, train_set, batch_size, config["use_cuda"])

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Compare losses and store model if better
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), model_name)
        train_losses.append(train_loss)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train loss: {train_loss:0.2}')
        print(f'\tTrain Loss: {train_loss:.3f}')
    
    with open(f"{model_name.split('.')[0]}_losses.json", 'w+') as json_file:
        json.dump({"train_losses": train_losses}, json_file, indent=4)


if __name__ == "__main__":
    model_config = open_file("configs/init_config.json")

    train(model_config)
