from turtle import forward
import torch
import torch.nn as nn

from transformers import AutoModel, BertModel


class OurModel(nn.Module):
    
    def __init__(self, config):
        super(OurModel, self).__init__()

        # Slot encoder - BERT model
        self.slot_encoder = BertModel.from_pretrained(config["bert_model"])
        for param in self.slot_encoder.parameters():
            param.requires_grad = False
        
        # Utterance encoder - SpanBERT model
        self.utt_encoder = AutoModel.from_pretrained(config["span_bert_model"])
    
    def forward(self, slot, slot_attn_mask, utt, utt_attn_mask):
        # slot shape: [32, 1, 5]
        # slot_attn_mask shape: [32, 1, 5]
        # utt shape: [32, 1, 50]
        # utt_attn_mask shape: [32, 1, 50]

        # === Get a single BERT contextualised vector for a slot ===
        # Remove dimension 1 since BERT model expects [B(atch), N(umber of tokens)] shape
        slot = slot.squeeze(dim=1)
        slot_attn_mask = slot_attn_mask.squeeze(dim=1)
        slot_vector = self.slot_encoder(slot, attention_mask=slot_attn_mask)
        
        # BERT output for [CLS] summarization of the whole sequence (same as in SUMBT paper)
        # slot_vector shape: [32, 768]
        slot_vector = slot_vector.pooler_output

        # === Get SpanBERT contextualised vectors for each token in utterance ===
        # Remove dimension 1 since SpanBERT model expects [B(atch), N(umber of tokens)] shape
        utt = utt.squeeze(dim=1)
        utt_attn_mask = utt_attn_mask.squeeze(dim=1)
        utt_vectors = self.utt_encoder(utt, attention_mask=utt_attn_mask)
        
        # SpanBERT last hidden state output for each input token
        # utt_vectors shape: [32, 50, 768]
        utt_vectors = utt_vectors.last_hidden_state

        # === Apply MultiHead attention between slot vector and utterance's tokens
        # Query is slot vector, while keys and values are utterance's tokens vectors

        # === Use RNN to output utterance tagged using BIO format ===

