import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3 are the inputs from different modalities with shape [B, 1, 4096]
        query1, key1, value1 = self.query(x1), self.key(x1), self.value(x1)
        query2, key2, value2 = self.query(x2), self.key(x2), self.value(x2)
        query3, key3, value3 = self.query(x3), self.key(x3), self.value(x3)
        
        # Attention between modalities
        attn12 = self.softmax(torch.matmul(query1, key2.transpose(-1, -2)))
        attn13 = self.softmax(torch.matmul(query1, key3.transpose(-1, -2)))
        attn23 = self.softmax(torch.matmul(query2, key3.transpose(-1, -2)))
        
        # Cross-modal interaction
        x1_interacted = torch.matmul(attn12, value2) + torch.matmul(attn13, value3)
        x2_interacted = torch.matmul(attn12.transpose(-1, -2), value1) + torch.matmul(attn23, value3)
        x3_interacted = torch.matmul(attn13.transpose(-1, -2), value1) + torch.matmul(attn23.transpose(-1, -2), value2)
        
        return x1_interacted, x2_interacted, x3_interacted
    
class CyclicCrossModalAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(CyclicCrossModalAttention, self).__init__()
        

        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x1, x2, x3):
        """
        输入：
        - x1, x2, x3: 每个模态的输入特征，形状为 [B, T, input_dim]
        输出：
        - x1_out, x2_out, x3_out: 每个模态的输出特征，形状为 [B, T, input_dim]
        """
        
        x1 = x1.transpose(0, 1)  # [T, B, input_dim]
        x2 = x2.transpose(0, 1)  # [T, B, input_dim]
        x3 = x3.transpose(0, 1)  # [T, B, input_dim]
        
       
        attn_output1, _ = self.multihead_attention(x1, x2, x3)
        attn_output2, _ = self.multihead_attention(x2, x3, x1)
        attn_output3, _ = self.multihead_attention(x3, x1, x2)
        fusion = (attn_output1+attn_output2+attn_output3)/3

        x_output1 = self.layer_norm(x1 + 0.01*self.dropout(fusion))  
        x_output2 = self.layer_norm(x2 + 0.01*self.dropout(fusion))  
        x_output3 = self.layer_norm(x3 + 0.01*self.dropout(fusion))  
        
        x1_out = x_output1.transpose(0, 1)
        x2_out = x_output2.transpose(0, 1)
        x3_out = x_output3.transpose(0, 1)
        
        return x1_out, x2_out, x3_out

   
class DynamicWeighting(nn.Module):
    def __init__(self, input_dim):
        super(DynamicWeighting, self).__init__()
        self.gate = nn.Linear(input_dim * 3, 3)  
        self.sigmoid_gate = nn.Sigmoid() 
    def forward(self, CXR, ECG, Lab):
        combined_input = torch.cat([CXR, ECG, Lab], dim=-1)  # Shape: [B, 1, input_dim * 3]
        gate_scores = self.sigmoid_gate(self.gate(combined_input))  # Shape: [B, 1, 3]
        weighted_CXR = gate_scores[..., 0].unsqueeze(-1) * CXR 
        weighted_ECG = gate_scores[..., 1].unsqueeze(-1) * ECG  
        weighted_Lab = gate_scores[..., 2].unsqueeze(-1) * Lab  


        return weighted_CXR, weighted_ECG, weighted_Lab