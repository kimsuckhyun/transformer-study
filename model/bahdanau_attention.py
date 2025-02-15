import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)
        score = self.v_a(torch.tanh(self.W_a(hidden) + self.U_a(encoder_outputs)))
        attention_weights = F.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights.squeeze(-1)
