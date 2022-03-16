import pdb
import math

from turtle import forward
import torch
import torch.nn as nn

from transformers import AutoModel, BertModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention class where query is encoded 
    slot type and keys and values are encoded user's utterance.
    """
    
    def __init__(self, hid_dim, heads, dropout=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.heads = heads
        
        assert self.hid_dim % self.heads == 0
        self.d_k = self.hid_dim // self.heads
        
        self.W_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_v = nn.Linear(self.hid_dim, self.hid_dim)
        self.out = nn.Linear(self.hid_dim, self.hid_dim)
        
        self.dropout = self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        Q = self.W_q(Q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)  # query = [bs, num_heads, query_len, d_k]
        K = self.W_k(K).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)  # key   = [bs, num_heads, key_len, d_k]
        V = self.W_v(V).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)  # value = [bs, num_heads, value_len, d_k]
        
        # size = [bs, num_heads, query_len, key_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        self.attn_scores = torch.softmax(scores, dim=-1)
        X = torch.matmul(self.dropout(self.attn_scores), V)  # size = [bs, num_heads, query_len, key_len] = [32, 8, 1, 50]
        
        X = X.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dim)
        self.X = self.out(X)    # [32, 1, 768]
        return self.attn_scores, self.X   


class OurModel(nn.Module):
    
    def __init__(self, config, num_tags, device):
        super(OurModel, self).__init__()

        self.utt_max_len = config["utt_max_len"]
        self.heads = config["heads"]    # number of heads in multi-headed attention
        self.dropout_prob = config["dropout"]
        self.rnn_hidden_dim = config["rnn_hidden_dim"]
        self.rnn_input_size = config["rnn_input_size"]
        self.rnn_num_layers = config["rnn_num_layers"]
        self.num_tags = num_tags
        self.device = device

        # Slot encoder - BERT model
        # This encoder's parameters are not fine-tuned (Same as in SUMBT paper)
        self.slot_encoder = BertModel.from_pretrained(config["bert_model"])
        for param in self.slot_encoder.parameters():
            param.requires_grad = False
        
        # Utterance encoder - SpanBERT model
        self.utt_encoder = AutoModel.from_pretrained(config["span_bert_model"])

        # MultiHead attention component
        self.bert_output_dim = self.utt_encoder.config.hidden_size  # 768
        self.mha = MultiHeadAttention(self.bert_output_dim, self.heads, dropout=self.dropout_prob)

        # RNN for BIO tagging component
        self.transform_to_rnn_input = nn.Linear(2*self.bert_output_dim, self.rnn_input_size)
        self.sigmoid = nn.Sigmoid()
        self.rnn = nn.LSTM(input_size=self.rnn_input_size,
                            hidden_size=self.rnn_hidden_dim,
                            num_layers=self.rnn_num_layers,
                            dropout=self.dropout_prob,   
                            batch_first=True)
        self.init_parameter(self.rnn)
        
        self.transform_to_tags = nn.Linear(self.rnn_hidden_dim, self.num_tags)
        self.layer_norm = nn.LayerNorm(self.num_tags)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, slot, slot_attn_mask, utt, utt_attn_mask):
        # slot shape: [32, 1, 5]
        # slot_attn_mask shape: [32, 1, 5]
        # utt shape: [32, 1, 50]
        # utt_attn_mask shape: [32, 1, 50]

        batch_size = slot.shape[0]
        
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
        Q = slot_vector
        K = utt_vectors
        V = utt_vectors
        attn_scores, X, = self.mha(Q, K, V, mask=None)
        X = X.squeeze(dim=1) # X = [32, 768]
                
        # === Use RNN to output utterance tagged using BIO format ===
        # Outputs will contain the final output
        outputs = torch.zeros(batch_size, self.utt_max_len, self.num_tags).to(self.device)

        # set initial with zeros hidden and cell state of rnn
        h = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim).to(self.device)
        c = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim).to(self.device)

        for t in range(self.utt_max_len):
            # Create input for the LSTM
            input_token = utt_vectors[:, t, :]  # [32, 768]
            lstm_input = torch.cat((input_token, X), dim=1) # [32, 1536]
            lstm_input = self.sigmoid(self.transform_to_rnn_input(lstm_input)) # [32, rnn_input_size]
            lstm_input = lstm_input.unsqueeze(dim=1) # [32, 1, rnn_input_size]

            # LSTM step
            lstm_output, (h, c) = self.rnn(lstm_input, (h, c))
            lstm_output = self.layer_norm(self.transform_to_tags(self.dropout(lstm_output))) # [32, 1, num_tags]
            lstm_output = lstm_output.squeeze(dim=1)

            # Store the LSTM output
            outputs[:, t, :] = lstm_output
        
        outputs = outputs.permute(0, 2, 1) # [32, num_tags, 50]

        return outputs
                
    @staticmethod
    def init_parameter(module):
        torch.nn.init.xavier_normal_(module.weight_ih_l0)
        torch.nn.init.xavier_normal_(module.weight_hh_l0)
        torch.nn.init.constant_(module.bias_ih_l0, 0.0)
        torch.nn.init.constant_(module.bias_hh_l0, 0.0)
