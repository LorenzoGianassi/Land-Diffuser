from torch.utils.data import Dataset
import torch
import numpy as np
import os
import fnmatch

class Diffusion_Dataset(Dataset):

    def __init__(self, audios_dir, sequences_dir, templates_dir, start_idx, end_idx):

        self.audios_dir = audios_dir
        self.sequences_dir = sequences_dir
        self.templates_dir = templates_dir
        self.start_idx = start_idx
        self.end_idx = end_idx

        print('The talking dataset contains: ')
        print(self.end_idx - self.start_idx)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        
        idx = idx + self.start_idx

        audio = np.load(os.path.join(self.audios_dir, 'seq' + str(idx).zfill(3) + '.npy'))
        
        sequence = np.load(os.path.join(self.sequences_dir, 'seq' + str(idx).zfill(3) + '.npy'))
        
        sequence = np.reshape(sequence, (len(sequence), 68*3))
        
        template = np.load(os.path.join(self.templates_dir, 'seq' + str(idx).zfill(3) + '.npy')).reshape((68*3))
        
        audio = torch.Tensor(audio)
        
        sequence = torch.Tensor(sequence)
        
        template = torch.Tensor(template)
        
        
        sample = {'audio': audio, 
                  'sequence': sequence, 
                  'template': template,
                  #'disp': displacements
                  }

        return sample


