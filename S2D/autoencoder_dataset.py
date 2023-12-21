from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.io import loadmat
#import get_landmarks
import random

# Define a custom dataset class that inherits from PyTorch's Dataset class.
class autoencoder_dataset(Dataset):

    def __init__(self, template, neutral_root_dir, points_dataset, shapedata, normalization=True, dummy_node=True):
     
        self.shapedata = shapedata


        self.normalization = normalization

        self.dummy_node = dummy_node

    
        self.neutral_root_dir = neutral_root_dir

        self.points_dataset = points_dataset

        self.paths = np.load(os.path.join(neutral_root_dir, 'paths_' + points_dataset + '.npy'))

        self.paths_lands = np.load(os.path.join(neutral_root_dir, 'landmarks_test.npy'))

    def __len__(self):
        # Return the number of samples in the dataset.
        return len(self.paths)

    def __getitem__(self, idx):
        # Method to retrieve a specific sample from the dataset.

        # Get the file basename from the loaded paths.
        basename = self.paths[idx]

        # Get the corresponding landmarks basename.
        basename_landmarks = self.paths_lands[idx]

        # Load input vertices from a file.
        verts_input = np.load(os.path.join(self.neutral_root_dir, 'points_input', basename + '.npy'), allow_pickle=True)

        # If a target file exists, load target vertices.
        if os.path.isfile(os.path.join(self.neutral_root_dir, 'points_target', basename + '.npy')):
            verts_target = np.load(os.path.join(self.neutral_root_dir, 'points_target', basename + '.npy'), allow_pickle=True)
        else:
            verts_target = np.zeros(np.shape(verts_input))

        # Load neutral landmarks and target landmarks.
        landmarks_neutral = np.load(os.path.join(self.neutral_root_dir, 'landmarks_input', basename_landmarks + '.npy'), allow_pickle=True)
        landmarks = np.load(os.path.join(self.neutral_root_dir, 'landmarks_target', basename_landmarks + '.npy'), allow_pickle=True)

        # Calculate the difference between target and neutral landmarks.
        landmarks = landmarks - landmarks_neutral

        # Apply data normalization if enabled.
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init / self.shapedata.std
            verts_neutral = verts_neutral - self.shapedata.mean
            verts_neutral = verts_neutral / self.shapedata.std

        # Replace NaN values in input vertices with 0.0.
        verts_input[np.where(np.isnan(verts_input))] = 0.0

        # Convert vertices and landmarks to float32 data type.
        verts_input = verts_input.astype('float32')
        landmarks = landmarks.astype('float32')

        # If `dummy_node` is enabled, add a dummy node to the input vertices.
        if self.dummy_node:
            verts_ = np.zeros((verts_input.shape[0] + 1, verts_input.shape[1]), dtype=np.float32)
            verts_[:-1, :] = verts_input
            verts_input = verts_

        # Convert data to PyTorch tensors.
        verts_input = torch.Tensor(verts_input)
        landmarks = torch.Tensor(landmarks)

        # Create a dictionary representing the sample.
        sample = {'points': verts_input, 'landmarks': landmarks, 'points_target': verts_target}

        return sample
