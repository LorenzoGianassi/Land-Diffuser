import numpy as np
from transformers import Wav2Vec2Processor
import librosa
import os
import torch
from wav2vec import Wav2Vec2Model
import pickle
from vocaset  import Get_landmarks

data_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/vertices_npy'
audio_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/wav'

with open('/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/templates.pkl', 'rb') as fin:
    templates = pickle.load(fin, encoding='latin1')

audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# wav2vec 2.0 weights initialization
audio_encoder.feature_extractor._freeze_parameters()

j = 0
for audio in os.listdir(audio_path):
    audios = []
    frame = []
    frame_1 = []
    frame_2 = []
    actor = str(audio[:24])
    template = templates[actor]
    template_landmarks = Get_landmarks.get_landmarks(template)
    sentence = audio[:-4] + '.npy'
    print(j)
    if os.path.exists(os.path.join(data_path, sentence)):
        vertices = np.load(os.path.join(data_path, sentence))
        vertices = np.reshape(vertices, (len(vertices), 5023, 3))
        landmarks = []
        for k in range(len(vertices)):
           landmarks.append(Get_landmarks.get_landmarks(vertices[k]))
        speech_array, sampling_rate = librosa.load(os.path.join(audio_path, audio), sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        hidden_states = audio_encoder(audio_feature, frame_num=len(vertices)).last_hidden_state
        hidden_states = hidden_states.detach().numpy().squeeze(0)
        
        np.save('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/sequences/seq' + str(j).zfill(3) + '.npy', landmarks)
        np.save('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/audios/seq' + str(j).zfill(3) + '.npy', hidden_states)
        np.save('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Sequence_Dataset/templates/seq' + str(j).zfill(3) + '.npy', template_landmarks)
        j+=1
        
        
        
