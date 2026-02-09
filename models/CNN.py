import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
    """
    Use BiLSTM as the baseline model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.feature_type = getattr(configs, 'feature_type', 'curve')
        
        # Dynamic Input Channels
        if hasattr(configs, 'feature_type') and configs.feature_type == 'extracted_features':
            in_channels = configs.enc_in
            # Use 1D Conv for extracted features
            self.cnn1 = nn.Conv1d(in_channels=in_channels, out_channels=configs.d_model, kernel_size=5, padding=2)
            self.cnn2 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=5, padding=2)
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
            # Calculate output size after convolutions and pooling
            # L -> L/2 -> L/2/2 = L/4
            pooled_len = self.early_cycle_threshold // 4
            # Project to d_model dimension (consistent with curve mode)
            self.flatten_projection = nn.Linear(configs.d_model * pooled_len, configs.d_model)
        else:
            in_channels = 3
            # Use 2D Conv for curve data
            self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=configs.d_model, kernel_size=5)
            self.cnn2 = nn.Conv2d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=5)
            self.flatten = nn.Flatten(start_dim=2)
            self.flatten_projection = nn.Linear(1311, configs.d_model)
        
        self.head = nn.Linear(configs.d_model, 1)


    def classification(self, x_enc, curve_attn_mask, return_embedding):
        '''
        x_enc: [B, L, num_variables, fixed_length_of_curve] for curve mode
               [B, L, num_features] for extracted_features mode
        '''
        # Intra-cycle modelling
        if x_enc.dim() == 3 and self.feature_type == 'extracted_features':
            # CSV feature mode: [B, L, num_features] -> [B, num_features, L]
            B, L, num_features = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1)  # [B, num_features, L]
            
            # 1D Convolutions
            output = self.cnn1(x_enc)
            output = F.relu(output)
            output = self.pool(output)  # L -> L/2
            output = self.cnn2(output)
            output = F.relu(output)
            output = self.pool(output)  # L/2 -> L/4
            
            # Flatten: [B, d_model, L/4] -> [B, d_model * L/4]
            output = output.reshape(B, -1)
            # Project to d_model: [B, d_model * L/4] -> [B, d_model]
            embedding = self.flatten_projection(output)
            
        elif x_enc.dim() == 4:
            # PKL curve mode: [B, L, num_variables, fixed_length_of_curve]
            B, L, num_var = x_enc.shape[0], x_enc.shape[1], x_enc.shape[2]
            x_enc = x_enc.reshape(B, num_var, L, -1)  # [B, num_var, L, fixed_length]
            
            # 2D Convolutions
            output = self.cnn1(x_enc)
            output = F.relu(output)
            output = F.avg_pool2d(output, kernel_size=5, stride=2)
            output = self.cnn2(output)
            output = F.relu(output)
            output = F.avg_pool2d(output, kernel_size=5, stride=2)
            output = self.flatten(output)
            
            # Project to d_model: [B, flattened_size] -> [B, d_model]
            embedding = self.flatten_projection(output)
        else:
            raise ValueError(f"Unsupported input shape: {x_enc.shape} with feature_type: {self.feature_type}")
        
        # Final output: [B, d_model] -> [B, 1]
        output = self.head(embedding)
        if return_embedding:
            return output, embedding
        return output

    def forward(self,  cycle_curve_data, curve_attn_mask, return_embedding=False):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve] for curve mode
                             [B, L, num_features] for extracted_features mode
            curve_attn_mask: [B, L]
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        if return_embedding:
            dec_out, embedding = self.classification(cycle_curve_data, curve_attn_mask, return_embedding)
            return dec_out, embedding  # [B, N]
        else:
            dec_out = self.classification(cycle_curve_data, curve_attn_mask, return_embedding)
            return dec_out  # [B, N]
