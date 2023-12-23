import S2D.models as models
import S2D.spiral_utils as spiral_utils
import S2D.shape_data as shape_data
import S2D.autoencoder_dataset as autoencoder_dataset
import S2D.save_meshes as save_meshes
import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from my_test_funcs import test_autoencoder_dataloader
import torch

import Get_landmarks
from lstm_ddpm import LSTM_Diffusion

from transformers import Wav2Vec2Processor
import time
import os
import cv2
import tempfile
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
import pyrender
import trimesh
import glob
import librosa
#from S2L.wavlm import WavLMModel
from wav2vec import Wav2Vec2Model

def integer_to_one_hot_encoding(integer, num_classes):
    if integer < 0 or integer >= num_classes:
        raise ValueError("Integer value is out of range for one-hot encoding with the specified number of classes.")
    encoding = [0] * num_classes
    encoding[integer] = 1
    return encoding


class Diffusion:
    def __init__(self,args, noise_steps=100, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = args.device 
        #self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-large")
        #self.processor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        self.beta = self.prepare_noise_schedule().to(args.device)
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
    def sample(self, args, audio, model, actor_name):

        min = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/min.npy')
        max = np.load('/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Normalized_New_Landmarks_Dataset/values/max.npy')
        
        
        with open(args.template_file, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        actor_vertices = templates[actor_name]
    
        
        template = Get_landmarks.get_landmarks(actor_vertices)
        speech_array, sampling_rate = librosa.load(audio, sr=16000)
        audio_feature = np.squeeze(self.processor(speech_array, sampling_rate=sampling_rate).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        hidden_states = self.audio_encoder(audio_feature).last_hidden_state.to(self.device)
        norm_template =  -1 + (2*(template - min) / (max - min))
        norm_template = np.reshape(norm_template, (1, 68*3))
        norm_template = torch.tensor(norm_template).to(self.device)
        #print(hidden_states.shape)
        print(f"Sampling new audio....")
        model.eval()
        #label = torch.FloatTensor(integer_to_one_hot_encoding(label_dict[label], 5)).to(device=args.device)
        #intensity = torch.FloatTensor(integer_to_one_hot_encoding(int(intensity) -1, 3)).to(device=args.device)
        with torch.no_grad():                     
            x = torch.randn((1, hidden_states.shape[1], 68*3)).to(self.device).float()
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(1) * i).long().to(self.device)
                x = model(x.float(), hidden_states.float(), norm_template, t)
                
            seq_gen_reshaped = x.cpu().detach().numpy().squeeze(0)
        seq_gen_reshaped = ((np.reshape(seq_gen_reshaped, (seq_gen_reshaped.shape[0], 68, 3)) + 1) * (max - min) / 2 + min)
        
        return seq_gen_reshaped, actor_vertices, template
    
os.environ['PYOPENGL_PLATFORM'] = 'egl'
def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):

    background_black = False
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2],
                               bg_color=[255, 255, 255])  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path , out_fname, fps, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for i_frame in range(num_frames - 2):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('/usr/bin/ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

def generate_mesh_video(out_path, out_fname, meshes_path_fname, fps, audio_fname, template):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

    audio_fname = audio_fname


    uv_template_fname = template
    sequence_vertices = []
    f = None

    for frame_idx, mesh_fname in enumerate(sequence_fnames):
        frame = Mesh(filename=mesh_fname)
        sequence_vertices.append(frame.v)
        if f is None:
            f = frame.f

    template = Mesh(sequence_vertices[0], f)
    sequence_vertices = np.stack(sequence_vertices)
    render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')


def generate_landmarks(args, diffusion, model_path, audio_path, template_file, save_path):
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch']
    model_state_dict = checkpoint['autoencoder_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    # Now, you can use the loaded objects as needed
    # For example, you can load the model's state and optimizer's state:
    model = LSTM_Diffusion(args.hidden_size, args.n_layers).to(args.device)  # Instantiate your model class
    model.load_state_dict(model_state_dict)
    model.eval()

    landmarks, actor_vertices, actor_landmarks  = diffusion.sample(args, audio_path, model, args.template_name)

    if not os.path.exists(os.path.join(save_path, 'points_input')):
        os.makedirs(os.path.join(save_path, 'points_input'))

    if not os.path.exists(os.path.join(save_path, 'points_target')):
        os.makedirs(os.path.join(save_path, 'points_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_target')):
        os.makedirs(os.path.join(save_path, 'landmarks_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_input')):
        os.makedirs(os.path.join(save_path, 'landmarks_input'))

    for j in range(len(landmarks)):
                np.save(os.path.join(save_path, 'points_input', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'points_target', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks[j])
                np.save(os.path.join(save_path, 'landmarks_input', '{0:08}_frame'.format(j)), actor_landmarks)

    files = []

    for r, d, f in os.walk(os.path.join(save_path, 'points_input')):
                for file in f:
                    if '.npy' in file:
                        files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'paths_test.npy'), sorted(files))

    files = []
    for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'landmarks_test.npy'), sorted(files))
    print('Done')

def generate_meshes_from_landmarks(template_path, reference_mesh_path, landmarks_path, prediction_path, save_meshes_path, args):


    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                                     test_file=landmarks_path + '/test.npy',
                                     reference_mesh_file=reference_mesh_path,
                                     normalization=False,
                                     meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    with open(
            '/mnt/diskone-first/lgianassi/Demo_Only_Meshes/S2D/template/template/COMA_downsample/downsampling_matrices.pkl',
            'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
         range(len(M_verts_faces))]

    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    for i in range(len(ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    Adj, Trigs = spiral_utils.get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage='trimesh')

    spirals_np, spiral_sizes, spirals = spiral_utils.generate_spirals(step_sizes,
                                                                      M, Adj, Trigs,
                                                                      reference_points=reference_points,
                                                                      dilation=dilation, random=False,
                                                                      meshpackage='trimesh',
                                                                      counter_clockwise=True)

    sizes = [x.vertices.shape[0] for x in M]

    device = torch.device(args.device)

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)

    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]
    dataset_test = autoencoder_dataset.autoencoder_dataset(neutral_root_dir=landmarks_path, points_dataset='test',
                                                           shapedata=shapedata,
                                                           normalization=False, template=reference_mesh_path)

    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, num_workers=4)

    model = models.SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                     filters_dec=filter_sizes_dec,
                                     latent_size=nz,
                                     sizes=sizes,
                                     nbr_landmarks=nbr_landmarks,
                                     spiral_sizes=spiral_sizes,
                                     spirals=tspirals,
                                     D=tD, U=tU, device=device).to(device)


    checkpoint = torch.load(args.S2D, map_location=device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    predictions, inputs, lands, targets = test_autoencoder_dataloader(device, model, dataloader_test, shapedata)
    np.save(os.path.join(prediction_path, 'targets'), targets)
    np.save(os.path.join(prediction_path, 'predictions'), predictions)
    save_meshes.save_meshes(predictions, save_meshes_path, n_meshes=len(predictions), template_path=template_path)
    print('Done')


def main():
    parser = argparse.ArgumentParser(description='S2L+S2D: Speech-Driven 3D Talking heads')
    parser.add_argument("--landmarks_dim", type=int, default=68 * 3, help='number of landmarks - 68*3')
    parser.add_argument("--audio_feature_dim", type=int, default=768, help='768 for wav2vec')
    parser.add_argument("--feature_dim", type=int, default=16, help='64 for vocaset')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=float, default=256, help='hidden_size of the model')
    #parser.add_argument("--num_layers", type=int, default=5, help='number of S2L layers')
    parser.add_argument("--n_layers", type=float, default=5, help='n layers of the lstm')
    parser.add_argument("--S2L", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/Diffusion/Results/diffusion_lstm_vel_loss_test.pth.tar', help='path to the S2L model')
    parser.add_argument("--S2D", type=str, default='/mnt/diskone-first/lgianassi/Demo_Only_Meshes/Models/s2d.pth.tar', help='path to the S2D model')
    parser.add_argument("--template_file", type=str, default="/mnt/diskone-first/lgianassi/Demo_Only_Meshes/templates.pkl", help='faces to animate')
    parser.add_argument("--template_name", type=str, default="FaceTalk_170913_03279_TA", help='face to animate')
    #parser.add_argument("--audio_path", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/wav/FaceTalk_170725_00137_TA_sentence23.wav', help='audio to animate')
    parser.add_argument("--audio_path", type=str, default='/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/wav/FaceTalk_170725_00137_TA_sentence24.wav', help='audio to animate')
    
    #parser.add_argument("--audio_path", type=str, default='/mnt/diskone-first/lgianassi/Italian.wav', help='audio to animate')

    parser.add_argument("--save_path", type=str, default='velocity_loss_ddpm', help='path for results')
    parser.add_argument("--flame_template", type=str, default="/mnt/diskone-first/lgianassi/LSTM_Tesi/vocaset/flame_model/FLAME_sample.ply", help='template_path')
    parser.add_argument("--video_name", type=str, default="velocity_loss.mp4", help='name of the rendered video')
    parser.add_argument("--fps", type=int, default=60, help='frames per second')

    args = parser.parse_args()
    test_audio_path = args.audio_path
    save_path = args.save_path
    diffusion = Diffusion(args)

    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'Landmarks'))
    os.mkdir(os.path.join(save_path, 'Meshes'))
    os.mkdir(os.path.join(save_path, 'predicted_meshes'))

    model_path = args.S2L
    audio_path = test_audio_path

    save_landmarks_path = os.path.join(save_path, 'Landmarks')
    actors_file = args.template_file

    template_path = args.flame_template
    prediction_path = os.path.join(save_path, 'Meshes')
    save_path_meshes = os.path.join(save_path, 'predicted_meshes')

    print('Landmarks generation')
    start = time.time()
    generate_landmarks(args, diffusion, model_path, audio_path, actors_file, save_landmarks_path)

    print('Meshes Generation')

    generate_meshes_from_landmarks(template_path, template_path, save_landmarks_path, prediction_path, save_path_meshes, args)
    end = time.time()

    print(str(end - start) + ' Seconds')

    save_video_path = save_path

    print('Video Generation')
    generate_mesh_video(save_video_path,
                        args.video_name,
                        save_path_meshes,
                        args.fps,
                        audio_path,
                        template_path)
    print('done')

if __name__ == '__main__':
    main()
