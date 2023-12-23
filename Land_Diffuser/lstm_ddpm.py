import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

 
class LSTM_Diffusion(nn.Module):
    def __init__(self, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.noise_embedding = nn.Linear(68*3, self.hidden_size)
        self.audio_embedding = nn.Linear(768, self.hidden_size)
        
        self.LSTM = nn.LSTM(self.hidden_size*3, self.hidden_size, self.n_layers, bidirectional=True, batch_first=True, dropout=0.2)
    
        self.output_mapper = nn.Linear(self.hidden_size*2, 204)

        self.sequence_pos_encoder = PositionalEncoding(self.hidden_size, 0.2)

        self.embed_timestep = TimestepEmbedder(self.hidden_size, self.sequence_pos_encoder)

        nn.init.constant_(self.output_mapper.weight, 0)
        nn.init.constant_(self.output_mapper.bias, 0)
      
    def forward(self, noise_lands, audio, template, t):
        
        noise_lands_emb = self.noise_embedding(noise_lands)
        
        time_emb = self.embed_timestep(t)
        
        time_emb = time_emb.repeat(noise_lands.shape[1], 1, 1)
        time_emb = time_emb.permute(1, 0, 2)

        audio_emb = self.audio_embedding(audio)
       
        input = torch.cat([audio_emb, time_emb, noise_lands_emb], dim=2)
        
        out, _ = self.LSTM(input)
        
        out = self.output_mapper(out)

        return out + template     

