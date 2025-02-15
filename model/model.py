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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(hidden_size + embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x.unsqueeze(1))
        context, _ = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=-1)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        return self.fc(output.squeeze(1)), hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        encoder_outputs, hidden = self.encoder(src)
        outputs = torch.zeros(trg.size(0), trg.size(1), self.decoder.fc.out_features).to(trg.device)

        input_token = trg[:, 0]
        for t in range(1, trg.size(1)):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = output
            input_token = output.argmax(1)  # Greedy decoding
        return outputs
