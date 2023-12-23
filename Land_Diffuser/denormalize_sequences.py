import numpy as np
import os




def normalize(max, min,sequence_path,normalized_sequence_path):
    sequence = np.load(sequence_path)
    sequence = sequence.reshape(sequence.shape[0],68,3)
    sequence = (sequence + 1) * (max - min) / 2 + min
    np.save(os.path.join(normalized_sequence_path, 'normalized_sequence.npy'),sequence)

if __name__ == "__main__":

    min = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/min.npy')
    max = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/max.npy')

    sequence_path = "/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Examples_Final_Strategy_Bigger_Model_vel_loss/70/sequence.npy"
    normalized_sequence_path="/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Examples_Final_Strategy_Bigger_Model_vel_loss/70/"
    normalize(max, min,sequence_path,normalized_sequence_path)
