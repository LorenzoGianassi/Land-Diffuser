from S2D import train_funcs
from S2D import models
from S2D import spiral_utils
from S2D import shape_data
from S2D import autoencoder_dataset
from S2D import train_funcs
from S2D import save_meshes
import pickle
import trimesh
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
import os
from sklearn.metrics.pairwise import euclidean_distances
from my_test_funcs import test_autoencoder_dataloader



filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
nz = 16
ds_factors = [4, 4, 4, 4]
reference_points = [[3567, 4051, 4597]]
nbr_landmarks = 68
step_sizes = [2, 2, 1, 1, 1]
dilation = [2, 2, 1, 1, 1]
device_idx = 0
torch.cuda.get_device_name(device_idx)

template_path = '/home/federico/Scrivania/Emotional_Speech/Template/FLAME_sample.ply'
reference_mesh_file = '/home/federico/Scrivania/Emotional_Speech/Speech2Land/VOCASET_Actors_Mouth_Closed/FaceTalk_170809_00138_TA.ply'
#reference_mesh_file = '/home/federico/Scrivania/Emotional_Speech/VOCASET_Actors/FaceTalk_170809_00138_TA.ply'
#root_dir = '/home/federico/Scrivania/Emotional_Speech/TEST_S2D/Data_lstm/IdSplit/fold_1'
root_dir = '/home/federico/Scrivania/Emotional_Speech/TEST_Speech2land/Data_BILSTM_Spec'
prediction_path = '/home/federico/Scrivania/Emotional_Speech/TEST_S2D/'
save_path_meshes = '/home/federico/Scrivania/Emotional_Speech/TEST_S2D/predicted_meshes'
meshpackage = 'trimesh'

shapedata = shape_data.ShapeData(nVal=100,
                      test_file=root_dir + '/test.npy',
                      reference_mesh_file=reference_mesh_file,
                      normalization=False,
                      meshpackage=meshpackage, load_flag=False)

shapedata.n_vertex = 5023
shapedata.n_features = 3



with open('/home/federico/Scrivania/Emotional_Speech/S2D/template/template/COMA_downsample/downsampling_matrices.pkl', 'rb') as fp:
    downsampling_matrices = pickle.load(fp)

M_verts_faces = downsampling_matrices['M_verts_faces']
M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in range(len(M_verts_faces))]

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
                                                        reference_points = reference_points,
                                                        dilation = dilation, random = False,
                                                        meshpackage = 'trimesh',
                                                        counter_clockwise = True)

sizes = [x.vertices.shape[0] for x in M]

device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

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


device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
dataset_test = autoencoder_dataset.autoencoder_dataset(neutral_root_dir=root_dir, points_dataset='test',
                                   shapedata=shapedata,
                                   normalization=False, template=reference_mesh_file)

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

print('loading checkpoint from file %s'%('/home/federico/Scrivania/Emotional_Speech/S2D_Results/s2d_with_voca.pth.tar'))
checkpoint = torch.load('/home/federico/Scrivania/Emotional_Speech/S2D_Results/s2d_with_voca.pth.tar', map_location=device)
model.load_state_dict(checkpoint['autoencoder_state_dict'])

predictions, inputs, lands, targets = test_autoencoder_dataloader(device, model, dataloader_test, shapedata)
np.save(os.path.join(prediction_path, 'targets'), targets)
np.save(os.path.join(prediction_path, 'predictions'), predictions)
save_meshes.save_meshes(predictions, save_path_meshes, n_meshes=len(predictions), template_path=template_path)

prediction = predictions
prediction = prediction[:, :-1, :3] * 1000
print(np.shape(prediction))

gt = targets
gt = gt[:, :, :3] * 1000
print(np.shape(gt))
mean_err = np.mean(np.sqrt(np.sum((prediction - gt[:, :, :]) ** 2, axis=2)))
std_err = np.std(np.sqrt(np.sum((prediction - gt[:, :, :]) ** 2, axis=2)))
print('Our error for this fold is', mean_err)
print('Our std for this fold is', std_err)

# else:
#
#     predictions, _, _, _ = test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000)
#     np.save(os.path.join(prediction_path, 'predictions'), predictions)
#     save_meshes(predictions, args['save_path_Meshes'], n_meshes=30)
