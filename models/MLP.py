import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.flatten = nn.Flatten(start_dim=1)

        # Dynamic Input Dimension
        if hasattr(configs, 'feature_type') and configs.feature_type == 'extracted_features':
            # Feature Mode: [B, Cycles, Features]
            self.input_dim = configs.early_cycle_threshold * configs.enc_in
        else:
            # Legacy Mode: [B, Cycles, 3, 100] (or similar 4D structure flattened)
            # Original code: self.charge_discharge_length*self.early_cycle_threshold*3
            self.input_dim = self.charge_discharge_length * self.early_cycle_threshold * 3

        self.first_layer = nn.Linear(self.input_dim, self.d_model)
        self.hidden_layers = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(self.e_layers)])
        self.pred_head = nn.Linear(self.d_model, 1)




    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False):
        '''
        cycle_curve_data: [B, early_cycle, fixed_len, num_var] OR [B, early_cycle, features]
        curve_attn_mask: [B, early_cycle]
        '''
        # Flatten all dimensions except Batch
        B = cycle_curve_data.shape[0]
        cycle_curve_data = cycle_curve_data.reshape(B, -1)

        out = self.first_layer(cycle_curve_data)
        for i in range(self.e_layers):
            out = self.hidden_layers[i](out)
        preds = self.pred_head(out)
        if return_embedding:
            return preds, out
        return preds
