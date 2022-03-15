import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

from utils.util import open_file


class Restaurant8kDataset(Dataset):

    def __init__(self, data_path, tokenizer_model, slot_max_len, utt_max_len, include_all=False):
        self.include_all = include_all
        self.slot_max_len = slot_max_len
        self.utt_max_len = utt_max_len
        self.bio_to_id = {'B': 1, 'I': 2, 'O': 3, 'P': 0}
        self.id_to_bio = {1: 'B', 2: 'I', 3: 'O', 0: 'P'}
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)

        self._init_data(data_path)

    def __getitem__(self, index):
        example = self.data[index]
        slot = example["slot"]
        utterance = example["utterance"]
        start_index = example["start_index"]
        end_index = example["end_index"]
        slot_value = utterance[start_index:end_index] if start_index != -1 else None

        # === Tokenize slot ===
        encoded_slot = self.tokenizer(slot,
                                      padding = "max_length",
                                      max_length = self.slot_max_len,
                                      return_attention_mask = True, 
                                      return_tensors = "pt")
        encoded_slot_ids = encoded_slot["input_ids"]
        encoded_slot_attn_mask = encoded_slot["attention_mask"]

        # === Tokenize utterance ===
        encoded_utt = self.tokenizer(utterance,
                                     padding = "max_length",
                                     max_length = self.utt_max_len,
                                     return_attention_mask = True, 
                                     return_tensors = "pt")
        encoded_utt_ids = encoded_utt["input_ids"]
        encoded_utt_attn_mask = encoded_utt["attention_mask"]

        # === Tag utterance with BIO tags ===
        tokenized_utt_len = int((encoded_utt_ids == 0).nonzero(as_tuple=True)[1][0])
        bio_tags = torch.zeros((1, self.utt_max_len))
        bio_tags[:, :tokenized_utt_len] = self.bio_to_id['O']
        if slot_value:
            encoded_slot_value = self.tokenizer(slot_value, add_special_tokens=False, return_tensors="pt")
            first_value_token = int(encoded_slot_value["input_ids"][0][0])
            values_tokens_len = encoded_slot_value["input_ids"].shape[1]
            value_start_index = int((encoded_utt_ids == first_value_token).nonzero(as_tuple=True)[1][0])
            bio_tags[:, value_start_index] = self.bio_to_id['B']
            bio_tags[:, value_start_index+1:value_start_index+values_tokens_len] = self.bio_to_id['I']

        # === Return all encoded_slot, encoded_utterance, tagged_utterance ===
        return encoded_slot_ids, encoded_slot_attn_mask, encoded_utt_ids, encoded_utt_attn_mask, bio_tags

    def __len__(self):
        return self.num_examples
    
    def _init_data(self, dpath):
        self.data = []
        full_data = open_file(dpath)

        for raw_example in full_data:
            no_label = "labels" not in raw_example

            if not self.include_all and no_label:
                continue
            
            utterance = raw_example["userInput"]["text"]
            if no_label:
                self.data.append({"utterance": utterance, 
                                  "slot": "none",
                                  "start_index": -1,
                                  "end_index": -1})
            else:
                for label in raw_example["labels"]:
                    slot = label["slot"]
                    end_index = label["valueSpan"]["endIndex"]

                    # If the utterance starts with slot value, start index is implicit
                    if "startIndex" not in label["valueSpan"]:
                        start_index = 0
                    else:
                        start_index = label["valueSpan"]["startIndex"]
                    
                    self.data.append({"utterance": utterance,
                                      "slot": slot,
                                      "start_index": start_index, 
                                      "end_index": end_index})

        self.num_examples = len(self.data)
