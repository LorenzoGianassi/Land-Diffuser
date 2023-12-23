import os
import sequence_data_loader as dl
import trimesh
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import argparse
from transformers import Wav2Vec2Processor
import librosa
from wav2vec import Wav2Vec2Model
from tqdm import tqdm
import torch.nn.functional as F
from vocaset  import Get_landmarks
from lstm_ddpm import LSTM_Diffusion



class Diffusion:
    def __init__(self,args, noise_steps=100, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = args.device 
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        self.beta = self.prepare_noise_schedule().to(args.device)
        #self.beta = self.sigmoid_beta_schedule(self.noise_steps).to(args.device)
        #self.beta = self.cosine_beta_schedule(self.noise_steps).to(args.device)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    '''The prepare_noise_schedule method generates a noise schedule by linearly spacing
      the beta values between beta_start and beta_end into noise_steps intervals.'''
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    

    def sigmoid_beta_schedule(self, timesteps, s=0.008):
        """
        Sigmoid schedule inspired by cosine schedule
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.sigmoid(((x / timesteps) + s) / (1 + s) * 5)  # Using sigmoid function
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)  # Using clamp instead of clip
    
    '''The noise_face method takes an input 3d face landmarks x and a time step t.
       It calculates the square root of self.alpha_hat and 1 - self.alpha_hat at the given time step.
       It then generates Gaussian noise Ɛ with the same shape as x using torch.randn_like.
       The method returns the noise landmarks obtained by combining x and Ɛ with the square root factors.'''
    def noise_faces_sequence(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


    '''The sample_timesteps method generates n random time steps for sampling. 
       It uses torch.randint to generate random integers between 1 (inclusive) and self.noise_steps (exclusive).'''
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    '''The sample method is the main sampling function. 
       It takes a model and the number of audio n to sample.'''
    def sample(self, args, model, epoch, test=False, hidden_state=None, seq_num= None):

        min = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/min.npy')
        max = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/max.npy')
        
        if test is not True:
            speech_array, sampling_rate = librosa.load(args.sample_audio, sr=16000)
            audio_feature = np.squeeze(self.processor(speech_array, sampling_rate=sampling_rate).input_values)
            audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
            audio_feature = torch.FloatTensor(audio_feature)
            hidden_states = self.audio_encoder(audio_feature).last_hidden_state.to(self.device)
        elif hidden_state is not None:
            hidden_states=hidden_state

        template = trimesh.load(args.template, process=False)
        template = Get_landmarks.get_landmarks(template.vertices)
        norm_template =  -1 + (2*(template - min) / (max - min))
        norm_template = np.reshape(norm_template, (1, 68*3))
        norm_template = torch.tensor(norm_template).to(self.device)
        print(f"Sampling new audio....")
        model.eval()
        with torch.no_grad():                     
            x = torch.randn((1, hidden_states.shape[1], 68*3)).to(self.device).float()
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(1) * i).long().to(self.device)
                x = model(x.float(), hidden_states.float(), norm_template, t)
                
            seq_gen_reshaped = x.cpu().detach().numpy().squeeze(0)
        model.train()
        if test is not True:
                if not os.path.exists(os.path.join(args.save_path, str(epoch))):
                    os.mkdir(os.path.join(args.save_path, str(epoch)))
                for m in range(len(seq_gen_reshaped)):
                    mesh = trimesh.Trimesh(((np.reshape(seq_gen_reshaped[m], (68,3)) + 1) * (max - min) / 2 + min))
                    mesh.export(os.path.join(args.save_path, str(epoch), 'tst' + str(m).zfill(3) + '.ply'))

                np.save(os.path.join(args.save_path, str(epoch), 'sequence.npy'), seq_gen_reshaped)
                
                mesh = trimesh.Trimesh(((torch.randn((68,3)) + 1) * (max - min) / 2 + min))
                mesh.export(os.path.join(args.save_path, str(epoch), 'Pure_Noise.ply'))
        else:   
            if not os.path.exists(os.path.join(args.save_path, str(epoch) + 'test')):
                    os.mkdir(os.path.join(args.save_path, str(epoch) + 'test'))
            for m in range(len(seq_gen_reshaped)):
                    mesh = trimesh.Trimesh(((np.reshape(seq_gen_reshaped[m], (68,3)) + 1) * (max - min) / 2 + min))
                    mesh.export(os.path.join(args.save_path, str(epoch)+ 'test', 'tst' + str(m).zfill(3) + '.ply'))    
            np.save(os.path.join(args.save_path, str(epoch) + 'test', 'sequence' + str(seq_num) + '.npy'), seq_gen_reshaped)    
            mesh = trimesh.Trimesh(((torch.randn((68,3)) + 1) * (max - min) / 2 + min))
            mesh.export(os.path.join(args.save_path, str(epoch) + 'test', 'Pure_Noise.ply'))
        return x

class Weighted_Loss(nn.Module):
    def __init__(self, args):
        super(Weighted_Loss, self).__init__()
        self.weights = torch.zeros(204)
        self.weights[15:39] = 1
        self.weights[144:] = 1
        self.weights = self.weights.to(args.device)
        self.mse = nn.MSELoss(reduction='none')
        self.cos_sim = nn.CosineSimilarity(dim=3)

    def forward(self, predictions, target):
        rec_loss = torch.mean(self.mse(predictions, target))

        mouth_rec_loss = torch.sum(self.mse(predictions * self.weights, target * self.weights)) / (84 * predictions.shape[1])
        
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]

        vel_loss = torch.mean((self.mse(prediction_shift, target_shift)))

        len = predictions.shape[1]
        predictions = torch.reshape(predictions, (1, len, 68, 3))
        target = torch.reshape(target, (1, len, 68, 3))

        cos_dist = torch.mean((1 - self.cos_sim(predictions, target)))

        return   0.1 * rec_loss + 0.1 * mouth_rec_loss + 10 * vel_loss #+ 0.1 * cos_dist
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    
    dataset_train = dl.Diffusion_Dataset(args.dataset_audio_dir, 
                                         args.dataset_sequence_dir,
                                         args.dataset_template_dir,
                                         #args.dataset_disp_dir,
                                         0,
                                         350)
    
    dataset_validation = dl.Diffusion_Dataset(args.dataset_audio_dir, 
                                         args.dataset_sequence_dir,
                                         args.dataset_template_dir,
                                         #args.dataset_disp_dir,
                                         350,
                                         400)
    
    
    dataloader_train = DataLoader(dataset_train, batch_size=1,
                                 shuffle=True, num_workers=32, pin_memory=True)
    
    dataloader_validation = DataLoader(dataset_validation, batch_size=1,
                                 shuffle=True, num_workers=32, pin_memory=True)

    device = args.device
    
    model = LSTM_Diffusion(args.hidden_size, args.n_layers).to(device)

    num_params = count_parameters(model)
    print(f"Number of trainable parameters in the model: {num_params}")
 
    criterion = Weighted_Loss(args)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    diffusion = Diffusion(args)
    
    best_loss_training = 10
    best_loss_test = 10
    
    for epoch in range(args.epochs):
        model.train()
        tloss = 0
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for b, sample in pbar:
            optim.zero_grad()
            audio = sample['audio'].to(device)
            sequence = sample['sequence'].to(device)
            template = sample['template'].to(device)
            t = diffusion.sample_timesteps(audio.shape[0]).to(device)
            noise_sequence, _ = diffusion.noise_faces_sequence(sequence, t)
            pred_sequence = model.forward(noise_sequence.float(), audio, template, t)
            loss = criterion(pred_sequence, sequence)
            loss.backward()
            optim.step()
            tloss += loss
            pbar.set_description(
                "(Epoch {}) TRAIN LOSS {:.8f}".format((epoch + 1), tloss/(b+1)))
        
        
        if tloss/(b+1) < best_loss_training:
            best_loss_training = tloss/(b+1)
            torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(args.result_dir, 'diffusion_lstm_10_layer_train.pth.tar'))
        
        if epoch % 10 == 0:
            diffusion.sample(args, model, epoch)
            model.eval()
            vloss = 0
            with torch.no_grad():
                pbar = tqdm(enumerate(dataloader_validation), total=len(dataloader_validation))
                for b, sample in pbar:
                    audio = sample['audio'].to(device)
                    sequence = sample['sequence'].to(device)
                    template = sample['template'].to(device)
                    t = diffusion.sample_timesteps(audio.shape[0]).to(device)
                    noise_sequence, _ = diffusion.noise_faces_sequence(sequence, t)
                    pred_sequence = model.forward(noise_sequence.float(), audio, template, t)
                    loss = criterion(pred_sequence, sequence)
                    vloss += loss
                    pbar.set_description(
                        "(Epoch {}) TEST LOSS {:.8f}".format((epoch + 1), vloss/(b+1)))
            
            if vloss/(b+1) < best_loss_test:
                best_loss_test = vloss/(b+1)
                torch.save({'epoch': epoch,
                        'autoencoder_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, os.path.join(args.result_dir, 'diffusion_lstm_10_layer_test.pth.tar')) 







def test(args):

    #model_file_path = os.path.join(args.result_dir, 'diffusion_lstm_vel_loss_train.pth.tar')
    model_file_path = '/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Results/diffusion_lstm_10_layer.pth.tar'
    diffusion = Diffusion(args)

    # Load the saved data
    checkpoint = torch.load(model_file_path)

    # Extract the saved objects
    epoch = checkpoint['epoch']
    model_state_dict = checkpoint['autoencoder_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    # Now, you can use the loaded objects as needed
    # For example, you can load the model's state and optimizer's state:
    model = LSTM_Diffusion(args.hidden_size, args.n_layers).to(args.device)  # Instantiate your model class
    model.load_state_dict(model_state_dict)

    dataset_test = dl.Diffusion_Dataset(args.dataset_audio_dir, 
                                         args.dataset_sequence_dir,
                                         args.dataset_template_dir,
                                         #args.dataset_disp_dir,
                                         400,
                                         470)
    
    
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=True, num_workers=32, pin_memory=True)
    
    pbar = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
    seq_num = 0
    for b, sample in pbar:
        audio = sample['audio'].to(args.device)
        sequence = sample['sequence'].to(args.device)
        template = sample['template'].to(args.device)
        diffusion.sample(args, model, epoch,test=True,hidden_state=audio,seq_num=seq_num)
        seq_num += 1



def main():
    parser = argparse.ArgumentParser(description='3DTalkingDiffusers: Diffusion Model for 3d talking heads generation')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--hidden_size", type=float, default=256, help='hidden_size of the model')
    parser.add_argument("--n_layers", type=float, default=10, help='n layers of the lstm')
    parser.add_argument("--reference_mesh_file", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/D2D/template/flame_model/FLAME_sample.ply', help='path of the template')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_audio_dir", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/audios')
    parser.add_argument("--dataset_sequence_dir", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/sequences')
    parser.add_argument("--dataset_template_dir", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/templates')
    parser.add_argument("--dataset_disp_dir", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_Sequence_Dataset/displacements')
    parser.add_argument("--result_dir", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Results')
    parser.add_argument("--sample_audio", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/wav/FaceTalk_170725_00137_TA_sentence24.wav')
    parser.add_argument("--template", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/flame_model/FLAME_sample.ply')
    parser.add_argument("--save_path", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/10_layers')
    parser.add_argument("--flame_template", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/flame_model/flame_model/FLAME_sample.ply')
    parser.add_argument("--info", type=str, default='')
    
    

    args = parser.parse_args()

    train(args)
    #test(args)

if __name__ == "__main__":
    main()