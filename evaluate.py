#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
<Function of script>
"""

import os
import sys
import numpy as np
import pdb
import glob
from tqdm import tqdm
import pdb
import json
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, BertModel, BertTokenizer


from model.model import OurModel
from utils.util import open_file, set_seed, epoch_time
from data.restaurant_dataset import Restaurant8kDataset



class Classify(object):
    def __init__(self, config):
        self.config = config
    
    def load_test_data(self):
        self.test_set = Restaurant8kDataset(self.config["test_path"],
                                            self.config["bert_model"],
                                            self.config["slot_max_len"],
                                            self.config["utt_max_len"],
                                            self.config["include_all"])
        print(f"Number of test examples: {len(self.test_set)}")

    def load_model(self):
        # Model initialization
        device = torch.device('cuda' if self.config["use_cuda"] else 'cpu')
        model_name = self.config["model_name"]
        
        # Load model
        self.model = OurModel(self.config, self.test_set.num_of_bio_tags(), device).to(device)
        self.model.load_state_dict(torch.load(model_name, map_location=device))
        print("Model loaded")
        
        # Load tokenizer model
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model"])
        
    def predict(self):
        self.model.eval()
        batch_size = self.config["batch_size"]
        data_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)
        
        # Start evaluation phase
        id = 0
        self.true_vs_pred_dict = {}
        actual = []
        predicted = []
        use_cuda = self.config["use_cuda"]
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # Initialize a batch
                encoded_slots    = batch[0].cuda() if use_cuda else batch[0]
                slots_attn_masks = batch[1].cuda() if use_cuda else batch[1]
                encoded_utts     = batch[2].cuda() if use_cuda else batch[2]
                utts_attn_masks  = batch[3].cuda() if use_cuda else batch[3]
                true_labels      = batch[4].cuda() if use_cuda else batch[4]
                true_labels      = true_labels.squeeze(dim=1) # [32, 50]
                
                pred_probs = self.model(encoded_slots, slots_attn_masks, encoded_utts, utts_attn_masks)
                pred_labels = torch.max(pred_probs, 1)[1].view(true_labels.size()).data
                predicted.extend(pred_labels.cpu().detach().numpy())
                actual.extend(true_labels.cpu().detach().numpy())
                
                for enc_slot, enc_utt, true_lab, pred_lab in zip(encoded_slots, encoded_utts, true_labels, pred_labels):
                    # first decode then detokenize using bert base.
                    enc_utt = enc_utt[0].detach().cpu().numpy()
                    enc_slot = enc_slot[0].detach().cpu().numpy()
                    enc_utt_ids = self.tokenizer.convert_ids_to_tokens(enc_utt)
                    dec_utt = self.tokenizer.decode(enc_utt, skip_special_tokens=True)
                    dec_slot =  self.tokenizer.decode(enc_slot, skip_special_tokens=True)
                    
                    # passing the id only to debug certain examples, that are missing predicted labels
                    id += 1
                    dec_true_label, true_value = self._id2bio(true_lab.detach().cpu().numpy().tolist(), enc_utt_ids, id)
                    dec_pred_label, pred_value = self._id2bio(pred_lab.detach().cpu().numpy().tolist(), enc_utt_ids, id)
                    
                    text = "Example: " + str(id)
                    self.true_vs_pred_dict[text] = {
                        "Text": dec_utt,
                        "Slot": dec_slot,
                        "True Value": true_value,
                        "True Label": dec_true_label,
                        "Pred Value": pred_value,
                        "Pred Label": dec_pred_label
                    }
                    # print(dec_utt)
        
        # Compute Prec, Rec, F1 for entire batch
        # print('F1 = {}'.format(self._f1_metric(actual, predicted)))
        with open(f"results/test_prediction.json", 'w+') as json_file:
            json.dump(self.true_vs_pred_dict, json_file, indent=4)
        

    def _id2bio(self, id_list, utt, example_id):
        # Tag2id mappings
        bio_to_id = {'B': 1, 'I': 2, 'O': 3, 'P': 0}
        id_to_bio = {1: 'B', 2: 'I', 3: 'O', 0: 'P'}
        
        bio_id = []
        for id in id_list:
            bio_id.append(id_to_bio[id])
        

        special_tokens={'unk_token': '[UNK]', 
                        'sep_token': '[SEP]', 
                        'pad_token': '[PAD]', 
                        'cls_token': '[CLS]', 
                        'mask_token': '[MASK]'}
        id_utt = list(map(list, zip(bio_id, utt)))
        
        # start debugging a specific example with unique id as seen in the json file
        # E.g. Example: 7
        # if example_id == 7:
        #     pdb.set_trace()
        
        bio_label = []
        value = []
        for tup in id_utt:
            if tup[1] not in special_tokens.values():
                if tup[0] in ["B", "I"]: # Start of span and inside
                    bio_label.append(tup[0])
                    value.append(tup[1])

        return " ".join(bio_label),  " ".join(value)
    
    def _f1_metric(self, pred, true):
        return f1_score(true, pred)
        

def main():
    """ main method """
    config = open_file("configs/init_config.json")
    clf = Classify(config)
    clf.load_test_data()
    clf.load_model()
    clf.predict()
    

if __name__ == "__main__":
    main()
    

# Example of incorrect slot filling: (Quoted from Example: 7 in train set)
# Actual above; predicte below
# [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# 
# 
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# 
# 
# [['O', '[CLS]'], ['O', 'Can'], ['O', 'i'], ['O', 'ch'], ['O', '##nage'], ['O', 'my'], ['O', 'booking'], ['O', 'from'], ['O', '18'], ['O', ':'], ['O', '00'], ['O', 'for'], ['O', '3'], ['O', 'people'], ['O', 'to'], ['B', '17'], ['I', ':'], ['I', '45'], ['O', 'for'], ['O', '4'], ['O', 'people'], ['O', '?'], ['O', '[SEP]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]'], ['P', '[PAD]']]
# [['O', '[CLS]'], ['O', 'Can'], ['O', 'i'], ['O', 'ch'], ['O', '##nage'], ['O', 'my'], ['O', 'booking'], ['O', 'from'], ['O', '18'], ['O', ':'], ['O', '00'], ['O', 'for'], ['O', '3'], ['O', 'people'], ['O', 'to'], ['O', '17'], ['O', ':'], ['O', '45'], ['O', 'for'], ['O', '4'], ['O', 'people'], ['O', '?'], ['O', '[SEP]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]'], ['O', '[PAD]']]
