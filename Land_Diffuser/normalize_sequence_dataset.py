import numpy as np
import os
import trimesh
import tqdm

old_audio_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/audios'
old_sequences_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/sequences'
old_template_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/templates'


new_audio_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/audios'
new_sequences_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/sequences'
new_template_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/templates'


min = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/min.npy')
max = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/max.npy')

for file in os.listdir(old_sequences_path):
    sequence = np.load(os.path.join(old_sequences_path, file))
    sequence = -1 + (2 * (sequence - min) / (max - min))
    np.save(os.path.join(new_sequences_path, file), sequence)

for file in os.listdir(old_template_path):
    frame = np.load(os.path.join(old_template_path, file))
    frame = -1 + (2 * (frame - min) / (max - min))
    np.save(os.path.join(new_template_path, file), frame) 
    
for file in os.listdir(old_audio_path):
    audio = np.load(os.path.join(old_audio_path, file))
    np.save(os.path.join(new_audio_path, file), audio)    
 


    
    