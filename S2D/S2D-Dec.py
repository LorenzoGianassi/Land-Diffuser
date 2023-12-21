import numpy as np
import json
import os
import copy
import _pickle as pickle

import mesh_sampling
import trimesh
from shape_data import ShapeData
from Get_landmarks import get_landmarks
from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader
from spiral_utils import get_adj_trigs, generate_spirals
from models import SpiralAutoencoder
from my_test_funcs import test_autoencoder_dataloader 
import torch
#from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import euclidean_distances
from save_meshes import save_meshes
#from PIL import Image
import glob
import argparse

parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--test_fold', type=int, help='Change the fold here from 1 to 4', default=1)
parser.add_argument('--Split', type=str, help='Expressions split protocol: Split=Expr, Id split: Split=Id', default='Id')
parser.add_argument('--Lands', type=str, help='Lands=GT if you want to reproduce table 1 results, Lands=Motion3DGAN if you want to show Motion3DGAN generated samples', default='GT')
parser.add_argument('--label', type=str, help='the desired label for Motion3DGAN samples', default='0')
args = parser.parse_args()

Split=args.Split
fold=args.test_fold
Lands=args.Lands

if Lands=='GT':
    root_dir = './Data_lstm/'+ Split +'Split/fold_' + str(fold)
    results_dir = './Models/' + Split +'Split/fold_' + str(fold)
    testresults_dir = './Results/' + Split +'Split/fold_' + str(fold)
else:
    root_dir = './Data_lstm/Motion3DGAN/sample_' + str(args.label)
    results_dir = './Models/' + Split + 'Split/fold_' + str(fold)
    testresults_dir = './Results/Motion3DGAN/sample_' + str(args.label)




template_dir = './template/'
meshpackage = 'trimesh'
GPU = True
device_idx = 0
torch.cuda.get_device_name(device_idx)

#############################################

args = {}
downsample_method = 'COMA_downsample'

reference_mesh_file = os.path.join(template_dir, 'template', 'template.obj')
downsample_directory = os.path.join(template_dir, 'template', downsample_method)
ds_factors = [4, 4, 4, 4]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
dilation_flag = True
if dilation_flag:
    dilation = [2, 2, 1, 1, 1]
else:
    dilation = None
reference_points = [[3567, 4051, 4597]]

args = { 'neutral_data': os.path.join(root_dir),
        'results_folder': os.path.join(results_dir),
        'testresults_folder': os.path.join(testresults_dir),
        'save_path_Meshes': os.path.join(testresults_dir,'predicted_meshes'),
        'save_path_animations': './Results/Gifs/',
        'reference_mesh_file': reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed': 2, 'loss': 'l1',
        'batch_size': 16, 'num_epochs': 300, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': 16,
        'ds_factors': ds_factors, 'step_sizes': step_sizes, 'dilation': dilation,
        'lr': 1e-3,
        'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99, 'decay_steps': 1,
        'resume': False, 'nbr_landmarks': 68,
        'shuffle': True, 'nVal': 100, 'normalization': False}

#args['results_folder'] = os.path.join(args['results_folder'])

if not os.path.exists(os.path.join(args['results_folder'])):
    os.makedirs(os.path.join(args['results_folder']))

checkpoint_path = os.path.join(args['results_folder'])
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

prediction_path = os.path.join(args['testresults_folder'], 'predictions')
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

if not os.path.exists(args['save_path_Meshes']):
    os.makedirs(args['save_path_Meshes'])

if not os.path.exists(downsample_directory):
    os.makedirs(downsample_directory)

#####################################################
if __name__ == "__main__":
    np.random.seed(args['seed'])
    print("Loading data .. ")
    shapedata = ShapeData(nVal=args['nVal'],
                          test_file=args['neutral_data'] + '/test.npy',
                          reference_mesh_file=args['reference_mesh_file'],
                          normalization=args['normalization'],
                          meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5024
    shapedata.n_features = 3

    print("Loading Transform Matrices ..")
    with open(os.path.join(args['downsample_directory'], 'downsampling_matrices.pkl'), 'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    if shapedata.meshpackage == 'mpi-mesh':
        M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
    elif shapedata.meshpackage == 'trimesh':
        M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
             range(len(M_verts_faces))]
    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    print("Calculating reference points for downsampled versions..")
    for i in range(len(args['ds_factors'])):
        if shapedata.meshpackage == 'mpi-mesh':
            dist = euclidean_distances(M[i + 1].v, M[0].v[reference_points[0]])
        elif shapedata.meshpackage == 'trimesh':
            dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    #####################################################

    sizes = [x.vertices.shape[0] for x in M]
    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage = shapedata.meshpackage)

    spirals_np, spiral_sizes,spirals = generate_spirals(args['step_sizes'],
                                                        M, Adj, Trigs,
                                                        reference_points = reference_points,
                                                        dilation = args['dilation'], random = False,
                                                        meshpackage = shapedata.meshpackage,
                                                        counter_clockwise = True)

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
        d[0,:-1,:-1] = D[i].todense()
        u[0,:-1,:-1] = U[i].todense()
        d[0,-1,-1] = 1
        u[0,-1,-1] = 1
        bD.append(d)
        bU.append(u)

    ###################################################

    torch.manual_seed(args['seed'])

    if GPU:
        device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]

    #################################################

    dataset_test = autoencoder_dataset(neutral_root_dir=args['neutral_data'], points_dataset='test',
                                       shapedata=shapedata,
                                       normalization = args['normalization'], template = reference_mesh_file)

    dataloader_test = DataLoader(dataset_test, batch_size=args['batch_size'],
                                 shuffle=False, num_workers=args['num_workers'])

    model = SpiralAutoencoder(filters_enc=args['filter_sizes_enc'],
                              filters_dec=args['filter_sizes_dec'],
                              latent_size=args['nz'],
                              sizes=sizes,
                              nbr_landmarks=args['nbr_landmarks'],
                              spiral_sizes=spiral_sizes,
                              spirals=tspirals,
                              D=tD, U=tU, device=device).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['regularization'])
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'], gamma=args['decay_rate'])
    else:
        scheduler = None

    if args['loss'] == 'l1':
        def loss_l1(outputs, targets, inputs, displacement):
            weights = np.load(template_dir + './template/Normalized_d_weights.npy')
            Weigths = torch.from_numpy(weights).float().to(device)
            target_expression = outputs - inputs
            L = (torch.matmul(Weigths, torch.abs(outputs - targets))).mean() + 0.1 * torch.abs(
                target_expression - displacement).mean()
            return L

        loss_fn = loss_l1
    #########################################################

    ## Testing
    print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar')))
    checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

    if Lands=='GT':
        predictions, inputs, lands, targets = test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant = 1000)
        np.save(os.path.join(prediction_path, 'targets'), targets)
        np.save(os.path.join(prediction_path,'predictions'), predictions)
        save_meshes(predictions, args['save_path_Meshes'], n_meshes=30)

        prediction = predictions
        prediction = prediction[:, :-1, :3] * 1000
        print(np.shape(prediction))

        gt = targets
        gt = gt[:, :, :3] * 1000
        print(np.shape(gt))
        mean_err = np.mean(np.sqrt(np.sum((prediction - gt) ** 2, axis=2)))
        std_err = np.std(np.sqrt(np.sum((prediction - gt) ** 2, axis=2)))
        print('Our error for this fold is', mean_err)
        print('Our std for this fold is', std_err)
    else:
        predictions, _, _, _ = test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000)
        np.save(os.path.join(prediction_path, 'predictions'), predictions)
        save_meshes(predictions, args['save_path_Meshes'], n_meshes=30)





