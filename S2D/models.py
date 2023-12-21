# Import necessary libraries and modules
import torch
import torch.nn as nn
import pdb  # Debugger module

# Define a custom PyTorch module for Spiral Convolution
class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True, device=None):
        # Constructor for the SpiralConv class
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        # Create a linear layer for convolution
        self.conv = nn.Linear(in_c * spiral_size, out_c, bias=bias)

        # Choose the activation function based on the input
        # Supported activations: 'relu', 'elu', 'leaky_relu', 'sigmoid', 'tanh', 'identity'
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        # Forward pass through the SpiralConv module
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        # Flatten the indices for selecting spiral vertices
        spirals_index = spiral_adj.view(bsize * num_pts * spiral_size)

        # Create batch index for selecting data from x
        batch_index = torch.arange(bsize, device=self.device).view(-1, 1).repeat([1, num_pts * spiral_size]).view(-1).long()

        # Select the vertices based on the spiral adjacency
        spirals = x[batch_index, spirals_index, :].view(bsize * num_pts, spiral_size * feats)

        # Apply convolution and activation
        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        # Reshape the output and apply zero padding to the last row
        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

# Define a PyTorch module for a Spiral Autoencoder
class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, nbr_landmarks, sizes, spiral_sizes, spirals, D, U, device, activation='elu'):
        # Constructor for the SpiralAutoencoder class
        super(SpiralAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        self.nbr_landmarks = nbr_landmarks

        self.conv = []  # List to store SpiralConv layers

        # Initialize a linear layer for latent decoding
        self.fc_latent_dec = nn.Linear(nbr_landmarks * 3, (sizes[-1] + 1) * filters_dec[0][0])

        self.dconv = []  # List to store SpiralConv layers for decoding
        input_size = filters_dec[0][0]

        # Iterate through the layers of the decoder
        for i in range(len(spiral_sizes) - 1):
            if i != len(spiral_sizes) - 2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i + 1]

                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i + 1]
            else:
                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i + 1]
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[1][i + 1]
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i + 1]

        self.dconv = nn.ModuleList(self.dconv)  # Convert the list to a ModuleList

    # Define the encoding step of the autoencoder
    def encode(self, x):
        # Encoding step: Process the input data
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        X = []

        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
            X.append(x)
        x = x.view(bsize, -1)
        return self.fc_latent_enc(x), X

    # Define the decoding step of the autoencoder
    def decode(self, z):
        # Decoding step: Generate reconstructed data
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        X = []

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)
        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = torch.matmul(U[-1 - i], x)
            x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i + 1]:
                x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
                j += 1
            X.append(x)
        return x, X

    def forward(self, x, landmarks):
        # Forward pass through the entire autoencoder
        landmarks = landmarks.view(landmarks.size()[0], landmarks.size()[1] * landmarks.size()[2])
        landmarks = landmarks.view(landmarks.size()[0], -1)
        X, X_dec = self.decode(landmarks)
        X_ = X + x
        return X_, X
    

    def predict(self, x, landmarks):
        landmarks = landmarks.unsqueeze(0)
        landmarks=landmarks.view(landmarks.size()[0], landmarks.size()[1]*landmarks.size()[2])
        landmarks = landmarks.view(landmarks.size()[0], -1)
        X, X_dec = self.decode(landmarks)
        X_ = X.squeeze(0)[1:, :] + x
        return X_
