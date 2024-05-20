#### Copy of train loop except loads both networks from directuv and runs ONLY inference pipeline
import warnings

import igl
import matplotlib
import numpy as np
import numpy.random
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import DeformationEncoder
from losses import UVLoss, symmetricdirichlet
from results_saving_scripts import save_mesh_with_uv
from DeformationDataset import DeformationDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import PerCentroidBatchMaker
import MeshProcessor
from utils import dclamp, ZeroNanGrad, order_edges_and_cut

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from signal import SIGUSR1

from meshing.mesh import Mesh
from meshing.analysis import computeFacetoEdges

USE_CUPY = True
has_gpu = "cpu"
if USE_CUPY and torch.cuda.is_available():
    import cupy
    has_gpu="gpu"

import math
from pytorch_lightning.loggers import TensorBoardLogger
from results_saving_scripts.plot_uv import plot_uv, export_views
from pathlib import Path
from results_saving_scripts import paper_stats
import json

FREQUENCY = 10 # frequency of logguing - every FREQUENCY iteration step
UNIT_TEST_POISSON_SOLVE = True

class MyNet(pl.LightningModule):
    '''
    the network
    '''
    def __init__(self, encoder, code_dim, args, point_dim=6, verbose=False, face_dim=0, n_edges=None):
        print("********** Some Network info...")
        print(f"********** code dim: {code_dim}")
        print(f"********** centroid dim: {point_dim}")
        super().__init__()
        self.args = args
        self.arch = self.args.arch

        # HACK: lossgt
        if not hasattr(args, "lossgt"):
            args.lossgt = False

        # HACK: intersectionloss
        if not hasattr(args, "intersectionloss"):
            args.intersectionloss = False

        # HACK: fftextrinsic
        if not hasattr(args, "fftextrinsic"):
            args.fftextrinsic = 0

        # HACK: gtedgeloss
        if not hasattr(args, "gtedgeloss"):
            args.gtedgeloss = None

        self.lossfcn = UVLoss(args, self.device)

        # NOTE: code dim refers to the pointnet encoding. Point_dim is centroid position (also potentially fourier features)
        layer_normalization = self.get_layer_normalization_type()

        if self.arch == "directuv":
            # Arch for direct UV prediction (assuming input topology)
            # diffusionnet + MLP for UVs
            vertex_input_dim = point_dim + code_dim

            # NOTE: Must use the gt topology for this diffusionnet
            from diffusionnet import DiffusionNet
            self.vertexencoder = DiffusionNet(C_in=vertex_input_dim, C_out=self.args.vertexdim, C_width=128, N_block=4, outputs_at='vertices',
                                            with_gradient_features=True, with_gradient_rotations=True)
            encoderdim = self.args.vertexdim

            if self.args.init:
                if self.args.fft > 0:
                    from utils import FourierFeatureTransform
                    self.fft = FourierFeatureTransform(2, self.args.fft, self.args.fftscale)
                    encoderdim += 2 * self.args.fft + 2
                else:
                    encoderdim += 2

            self.uv_mlp = nn.Sequential(nn.Linear(encoderdim, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 2),
                                        )

        elif self.arch == "directedge":
            # MeshCNN + MLP for edge weights
            from source_njf.models import MeshEncoderDecoder

            edge_input_dim = 2 * (point_dim + code_dim) # Symmetric concat

            # NOTE: Mesh input = list(meshes)
            ncf = [64, 64, 128, 256]

            # HACK
            if not hasattr(self.args, "pool_res"):
                if "nopool" in self.args.expname:
                    pool_res = []
                else:
                    pool_res = [1800, 1350, 600]
            else:
                pool_res = self.args.pool_res

            fc_n = 64
            resblocks = 3
            embedding_size = 64
            num_heads = 8
            window_size = 35

            if self.args.initweightinput:
                if self.args.fftweight > 0:
                    from utils import FourierFeatureTransform

                    self.fftweight = FourierFeatureTransform(1, self.args.fftweight, self.args.fftscale)
                    edge_input_dim += 2 * self.args.fftweight + 1

                else:
                    edge_input_dim += 1

            ### Edge-based message passing: meshcnn + transformer
            # NOTE: outputs fc_n dim edge features
            from source_njf.models import get_norm_layer
            norm = "group"
            num_groups = 16
            norm_layer = get_norm_layer(norm_type=norm, num_groups=num_groups)

            down_convs = [edge_input_dim] + ncf
            up_convs = ncf[::-1] + [fc_n]

            pool_res = [n_edges] + pool_res
            self.edge_decoder = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=resblocks,
                                    transfer_data=len(pool_res) > 1)
            self.edgeweight_mlp = nn.Sequential(nn.Linear(fc_n, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1),
                                                )

        elif self.args.softpoisson:
            assert face_dim > 0, f"face_dim must be > 0 for soft poisson. face_dim: {face_dim}."

            # NOTE: Dimension of output depends on max of # edge weights being predicted across dataset
            output_dim = 9 + face_dim
            input_dim = point_dim + code_dim

            ## DiffusionNet vertex encoder
            if self.arch == "diffusionnet":
                from diffusionnet import DiffusionNet
                vertexdim = self.args.vertexdim

                # TODO: fourier features on extrinsic features
                self.vertexencoder = DiffusionNet(C_in=input_dim, C_out=self.args.vertexdim, C_width=128, N_block=4, outputs_at='vertices',
                                                with_gradient_features=True, with_gradient_rotations=True)
                face_decoder_dim = 9
                edge_decoder_dim = 1
                self.edge_decoder = nn.Sequential(nn.Linear(self.args.vertexdim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, edge_decoder_dim),
                                            )

                # Initialize edge weights to 0
                self.edge_decoder[-1].bias.data.zero_()
                self.edge_decoder[-1].weight.data.zero_()

                self.face_decoder = nn.Sequential(nn.Linear(self.args.vertexdim, 128),
                                            # nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                            nn.ReLU(),
                                            nn.Linear(128, 256),
                                            # nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                            nn.ReLU(),
                                            nn.Linear(256, 128),
                                            # nn.GroupNorm(num_groups=4, num_channels=128),
                                            nn.ReLU(),
                                            nn.Linear(128, face_decoder_dim),
                                            )

                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.face_decoder[-1].bias.data.zero_()
                    self.face_decoder[-1].weight.data.zero_()

            # Diffusionnet => vertex features => transfmeshcnn (edge features) + faceconv (face features) => jacobians + edge weights
            elif self.arch == "fullconv":
                from diffusionnet import DiffusionNet

                input_dim = point_dim + code_dim
                vertexdim = self.args.vertexdim

                if self.args.fftextrinsic and self.args.extrinsic:
                    from source_njf.utils import FourierFeatureTransform
                    self.fftextrinsic = FourierFeatureTransform(3, self.args.fftextrinsic, self.args.fftscale)
                    input_dim = input_dim + 2 * self.args.fftextrinsic + 6 # [input, sin, cos]

                self.vertexencoder = DiffusionNet(C_in=input_dim, C_out=vertexdim, C_width=128, N_block=4, outputs_at='vertices',
                                                with_gradient_features=True, with_gradient_rotations=True)

                from source_njf.models import MeshEncoderDecoder

                # NOTE: Mesh input = list(meshes)
                input_nc =  2 * vertexdim # NOTE: symmetric operation applied to features of each edge
                ncf = [64, 64, 128, 256]

                # HACK
                if not hasattr(self.args, "pool_res"):
                    if "nopool" in self.args.expname:
                        pool_res = []
                    else:
                        pool_res = [1800, 1350, 600]
                else:
                    pool_res = self.args.pool_res

                fc_n = 64
                resblocks = 3
                embedding_size = 64
                num_heads = 8
                window_size = 35

                if self.args.initweightinput:
                    if self.args.fftweight > 0:
                        from utils import FourierFeatureTransform

                        self.fftweight = FourierFeatureTransform(1, self.args.fftweight, self.args.fftscale)
                        input_nc += 2 * self.args.fftweight + 1

                    else:
                        input_nc += 1

                ### Edge-based message passing: meshcnn + transformer
                # NOTE: outputs fc_n dim edge features
                from source_njf.models import get_norm_layer
                norm = "group"
                num_groups = 16
                norm_layer = get_norm_layer(norm_type=norm, num_groups=num_groups)

                down_convs = [input_nc] + ncf
                up_convs = ncf[::-1] + [fc_n]

                pool_res = [n_edges] + pool_res
                self.edge_decoder = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=resblocks,
                                        transfer_data=len(pool_res) > 1)

                self.edgeweight_mlp = nn.Sequential(nn.Linear(fc_n, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 256),
                                                    nn.ReLU(),
                                                    nn.Linear(256, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 1),
                                                    )
                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.edgeweight_mlp[-1].bias.data.zero_()
                    self.edgeweight_mlp[-1].weight.data.zero_()

                # Face features are averaged vertex features
                faceinputdim = vertexdim
                faceoutput = 6

                if self.args.initjinput:
                    faceoutput = 4
                    if self.args.fftj > 0:
                        from utils import FourierFeatureTransform

                        self.fftj = FourierFeatureTransform(6, self.args.fftj, self.args.fftscale)
                        faceinputdim += 2 * self.args.fftj + 6

                    else:
                        faceinputdim += 6

                ## Face-based message passing: face convolutions from geometric texture synthesis
                from source_njf.geometrictextures.models.single_mesh_conv import MultiMeshConv

                fdim = self.args.facedim
                number_features = [faceinputdim, 128, 256, 128, fdim]
                self.face_decoder = MultiMeshConv(number_features)
                self.facejacobian_mlp = nn.Sequential(nn.Linear(fdim, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 256),
                                                    nn.ReLU(),
                                                    nn.Linear(256, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, faceoutput),
                                                    )

                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.facejacobian_mlp[-1].bias.data.zero_()
                    self.facejacobian_mlp[-1].weight.data.zero_()

            ## TODO
            elif self.arch == "attention":
                pass

            elif self.arch == "mlp":
                self.face_decoder = nn.Sequential(nn.Linear(input_dim, 128),
                                                    # nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    # nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    # nn.GroupNorm(num_groups=4, num_channels=128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 128),
                                                    # nn.GroupNorm(num_groups=4, num_channels=128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, output_dim))
                # Initialize the last layer sparsely
                torch.nn.init.sparse_(self.face_decoder[-1].weight.data[9:], 0.2)

                self.__IDENTITY_INIT = self.args.identity
                if self.__IDENTITY_INIT:
                    self.face_decoder[-1].bias.data[:].zero_()
                    self.face_decoder[-1].weight.data[:9].zero_()

            else:
                raise NotImplementedError(f"Unknown architecture: {self.arch}.")

        elif hasattr(self.args, "dense") and self.args.dense:
            print("==== We are predicting FLAT vectors! ==== ")
            if self.args.dense == "xyz":
                channels = (point_dim + code_dim) * 3
            else:
                channels = point_dim + code_dim
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(normalized_shape=channels),
                                                    nn.ReLU(),
                                                    nn.Linear(channels, channels),
                                                    )
        elif layer_normalization == "GROUPNORM":
            # print("Using GROUPNORM in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128), # , eps=0.0001 I have considered increasing this value in case we have channels from pointnet with the same values.
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.GroupNorm(num_groups=4, num_channels=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        elif layer_normalization == "LAYERNORM":
            # print("Using LAYERNORM in face_decoder!")
            self.face_decoder = nn.Sequential(nn.Linear(point_dim + code_dim, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.LayerNorm(normalized_shape=128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 9))
        else:
            raise Exception("unknown normalization method")

        self.__global_trans = self.args.globaltrans
        if self.__global_trans:
            self.global_decoder = nn.Sequential(nn.Linear(code_dim, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 9))

        self.encoder = encoder
        self.point_dim = point_dim
        self.code_dim = code_dim
        self.verbose = verbose
        self.mse = nn.MSELoss()
        self.save_hyperparameters()
        self.log_validate = True
        self.lr = args.lr
        self.val_step_iter = 0
        self.__test_stats = paper_stats.PaperStats()

        # For iterative reweighting
        self.stitchweights = {}

    ##################
    # inference code below
    ##################
    def forward(self, x, source=None):
        '''
		The MLP applied to a (batch) of global code concatenated to a centroid (z|c)
		:param x: B x (|z|+|c|) batch of (z|c) vectors
		:return: B x 9 batch of 9 values that are the 3x3 matrix predictions for each input vector
		'''
        if self.arch == 'mlp':
            return self.face_decoder(x)
        elif self.arch in ['diffusionnet', 'directuv']:
            return self.vertexencoder(x, source.get_loaded_data('mass'), L=source.get_loaded_data('L'),
                                    evals=source.get_loaded_data('evals'), evecs=source.get_loaded_data('evecs'),
                                    gradX=source.get_loaded_data('gradX'), gradY=source.get_loaded_data('gradY'),
                                    faces=source.get_loaded_data('faces'))
        elif self.arch == 'fullconv':
                # Check for fftextrinsics
                if self.args.fftextrinsic and self.args.extrinsic:
                    fftvertices = self.fftextrinsic(source.get_vertices().to(self.device))
                    normals = source.get_loaded_data('vertexnormals')
                    x = torch.cat([x, fftvertices, normals], dim=1)

                vertexcodes = self.vertexencoder(x, source.get_loaded_data('mass'), L=source.get_loaded_data('L'),
                                        evals=source.get_loaded_data('evals'), evecs=source.get_loaded_data('evecs'),
                                        gradX=source.get_loaded_data('gradX'), gradY=source.get_loaded_data('gradY'),
                                        faces=source.get_loaded_data('faces'))
                return vertexcodes

    def predict_jacobians(self, source, target, initj=None):
        '''
		given a batch class, predict jacobians
		:param single_source_batch: batch object
		:return: BxTx3x3 a tensor of 3x3 jacobians, per T tris, per B targets in batch
		'''
        # extract the encoding of the source and target
        if self.args.noencoder:
            codes = None
        else:
            codes = self.extract_code(source, target)

        # get the network predictions, a BxTx3x3 tensor of 3x3 jacobians, per T tri, per B target in batch
        return self.predict_jacobians_from_codes(codes, source, initj)

    def predict_jacobians_from_codes(self, codes, source, initj = None):
        '''
		predict jacobians w.r.t give global codes and the batch
		:param codes: codes for each source/target in batch
		:param single_source_batch: the batch
		:return:BxTx3x3 a tensor of 3x3 jacobians, per T tris, per B targets in batch
		'''
        if self.args.dense:
            stacked = source.flat_vector
        elif codes is None:
            stacked = source.get_input_features()
        else:
            # take all encodings z_i of targets, and all centroids c_j of triangles, and create a cartesian product of the two as a 2D tensor so each sample in it is a vector with rows (z_i|c_j)
            net_input = PerCentroidBatchMaker.PerCentroidBatchMaker(codes, source.get_input_features(), args=self.args)
            stacked = net_input.to_stacked()
            if self.args.layer_normalization != "GROUPNORM2":
                stacked = net_input.prep_for_linear_layer(stacked)
            else:
                stacked = net_input.prep_for_conv1d(stacked)

        # feed the 2D tensor through the network, and get a 3x3 matrix for each (z_i|c_j)
        res = self.forward(stacked, source)

        ##### SOFT POISSON #####
        if self.arch == 'diffusionnet':
            # Construct edge and face codes from vertex codes
            vertexcodes = res

            if self.args.inittopo:
                fuv_to_v = source.fuv_to_v
                vidxs, vcounts = np.unique(fuv_to_v, return_counts=True)
                new_vcodes = torch.zeros(len(vidxs), vertexcodes.shape[1]).to(vertexcodes.device)
                dup_vs = vidxs[np.where(vcounts > 1)[0]]
                keep_vs = vidxs[np.where(vcounts == 1)[0]]
                new_vcodes[keep_vs] = vertexcodes[keep_vs]
                for dup_v in dup_vs:
                    dup_vcodes = vertexcodes[fuv_to_v == dup_v]
                    new_vcodes[dup_v] = torch.mean(dup_vcodes, dim=0)
                vertexcodes = new_vcodes

            faces = source.get_source_triangles()
            edge_vpairs_nobound = source.ogedge_vpairs_nobound
            facecodes = torch.mean(vertexcodes[faces,:], dim=1)
            edgecodes = torch.mean(vertexcodes[edge_vpairs_nobound,:], dim=1)

            res = self.face_decoder(facecodes)
            edgevals = self.edge_decoder(edgecodes).squeeze()

            # Compute edge weights
            if self.args.spweight == "sigmoid":
                weights = -torch.sigmoid(edgevals + source.initweights.to(edgevals.device))
            # Softmax the weights
            elif self.args.spweight == "softmax":
                weights = torch.softmax(edgevals + source.initweights.to(edgevals.device), dim=0)

                # Rescale just to make viz easier (not necessary for soft poisson to work)
                weights = weights / torch.max(weights).detach()
                weights = -weights

            elif self.args.spweight == "nonzero":
                weights = -dclamp(edgevals + source.initweights.to(edgevals.device), 1e-9, 10000)

        elif self.arch == 'fullconv':
            # Construct edge and face codes from vertex codes
            vertexcodes = res

            if self.args.inittopo:
                fuv_to_v = source.fuv_to_v
                vidxs, vcounts = np.unique(fuv_to_v, return_counts=True)
                new_vcodes = torch.zeros(len(vidxs), vertexcodes.shape[1]).to(vertexcodes.device)
                dup_vs = vidxs[np.where(vcounts > 1)[0]]
                keep_vs = vidxs[np.where(vcounts == 1)[0]]
                new_vcodes[keep_vs] = vertexcodes[keep_vs]
                for dup_v in dup_vs:
                    dup_vcodes = vertexcodes[fuv_to_v == dup_v]
                    new_vcodes[dup_v] = torch.mean(dup_vcodes, dim=0)
                vertexcodes = new_vcodes

            ### Edge decoding
            #### Predict edge weights #####
            # NOTE: symmetric edge features (|v1 - v2|, v1 + v2)
            edge_vpairs_nobound = source.ogedge_vpairs_nobound.detach().cpu().numpy()

            # Vertex features => MeshCNN edge features
            meshcnn_edge_vpairs = source.edges
            edgecodes = torch.cat([torch.abs(vertexcodes[meshcnn_edge_vpairs[:,0]] - vertexcodes[meshcnn_edge_vpairs[:,1]]),
                                    vertexcodes[meshcnn_edge_vpairs[:,0]] + vertexcodes[meshcnn_edge_vpairs[:,1]]], dim=1)

            # NJF edges => MeshCNN edges
            edge2key = source.edge2key
            meshcnn_to_njfnobound = []
            for i, edge in enumerate(edge_vpairs_nobound):
                meshcnn_to_njfnobound.append(edge2key[tuple(sorted(edge))])
            meshcnn_to_njfnobound = np.array(meshcnn_to_njfnobound)

            if self.args.initweightinput:
                initweights = source.initweights.to(self.device).unsqueeze(1)

                if self.args.fftweight > 0:
                    initweights = self.fftweight(initweights)

                # NOTE: boundary weights will default to 0
                mapped_weights = torch.zeros(len(edgecodes), 1, dtype=initweights.dtype).to(self.device)
                mapped_weights[meshcnn_to_njfnobound] = initweights
                edgecodes = torch.cat([edgecodes, mapped_weights], dim=1)

            edgecodes = edgecodes.transpose(1, 0) # features x E
            edgevals = self.edge_decoder(edgecodes.unsqueeze(0), [source]).squeeze().transpose(1,0)
            edgevals = self.edgeweight_mlp(edgevals).squeeze()

            # MeshCNN edges => NJF edges
            edgevals = edgevals[meshcnn_to_njfnobound]

            # Compute edge weights
            if self.args.spweight == "sigmoid":
                weights = -torch.sigmoid(edgevals + source.initweights.to(edgevals.device))
            # Softmax the weights
            elif self.args.spweight == "softmax":
                weights = torch.softmax(edgevals + source.initweights.to(edgevals.device), dim=0)

                # Rescale just to make viz easier (not necessary for soft poisson to work)
                weights = weights / torch.max(weights).detach()
                weights = -weights

            elif self.args.spweight == "nonzero":
                weights = -dclamp(edgevals + source.initweights.to(edgevals.device), 1e-9, 10000)

            faces = source.get_source_triangles()
            edge_vpairs_nobound = source.ogedge_vpairs_nobound

            #### Predict jacobians #####
            facecodes = torch.mean(vertexcodes[faces,:], dim=1)

            # Conditioning
            if self.args.initjinput:
                # Remove the 0 dimension
                initjinput = initj[:, :2, :].reshape(len(facecodes), -1)

                if self.args.fftj > 0:
                    initjinput = self.fftj(initjinput)

                facecodes = torch.cat([facecodes, initjinput], dim=1)

            facecodes = facecodes.transpose(1, 0) # features x F
            res = self.face_decoder(facecodes, source.gfmm).squeeze().transpose(1,0)
            res = self.facejacobian_mlp(res)

            # Pad to 3x3
            if res.shape[-1] == 4:
                res = res.reshape(-1, 2, 2)
                res = torch.cat([res, torch.zeros(len(res), 1, 2).to(res.device)], dim=1)
                res = torch.cat([res, torch.zeros(len(res), 3, 1).to(res.device)], dim=2)

            elif res.shape[-1] == 6:
                res = res.reshape(-1, 3, 2)
                res = torch.cat([res, torch.zeros(len(res), 3, 1).to(res.device)], dim=2)

        # TODO: take SA-computed face latents => cosine similarity => dclamp(-2, 2) => add 1 div 2 (0 - 1 scaling)
        #                                     => face decoder MLP => jacobians
        elif self.arch == 'attention':
            pass

        elif self.arch == 'mlp':
            # NOTE: soft poisson weights will be dot product of "face latents" predicted per triangle for corresponding faces
            facelatents = res[:, 9:]
            facepairs = source.facepairs_nobound

            if self.args.spweight == "sigmoid":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                facesim = torch.sigmoid(facedot + source.initweights.to(facedot.device))
            elif self.args.spweight == "nonzero":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                weights = -dclamp(facedot + source.initweights.to(edgevals.device), 1e-7, 1)
            elif self.args.spweight == "cosine":
                facesim = dclamp(torch.nn.functional.cosine_similarity(facelatents[facepairs[:,0]], facelatents[facepairs[:,1]]) \
                                + source.initweights.to(facelatents.device), 1e-7, 1)
            elif self.args.spweight == "softmax":
                facedot = torch.sum(facelatents[facepairs[:,0]] * facelatents[facepairs[:,1]], dim=1)
                facesim = torch.softmax(facedot + source.initweights.to(facedot.device))

                # Rescale just to make viz easier (not necessary for soft poisson to work)
                with torch.no_grad():
                    weights = weights/torch.max(weights)
            else:
                raise Exception(f"Unknown soft poisson weight type: {self.args.spweight}.")
            weights = -facesim

            if self.args.softpoisson == "edges":
                assert len(weights) == len(source.edge_vpairs), f"weights len: {len(weights)}, edge vpairs len: {len(source.edge_vpairs)}"
            else:
                assert len(weights) == len(source.valid_pairs), f"weights len: {len(weights)}, valid pairs len: {len(source.valid_pairs)}"

            res = res[:, :9]

            if self.args.optweight:
                weights = -dclamp(self.trainer.optimizers[0].param_groups[1]['params'][0] + source.initweights.to(facelatents.device), 1e-7, 1)


        # No global codes
        if self.args.dense or codes is None:
            ret = res.reshape(1, source.mesh_processor.faces.shape[0], 3, 3)

            if self.__IDENTITY_INIT:
                for i in range(0, 3):
                    ret[:, :, i, i] += 1

            return ret, weights

        ## TODO: REFACTOR BELOW IF GLOBAL CODES ##
        # because of stacking the result is a 9-entry vec for each (z_i|c_j), now let's turn it to a batch x tris x 9 tensor
        pred_J = net_input.back_to_non_stacked(res)
        # and now reshape 9 to 3x3
        ret = pred_J.reshape(pred_J.shape[0], pred_J.shape[1], 3, 3)
        # if we apply a global transformation
        if self.__global_trans:
            glob = self.global_decoder(codes)
            glob = glob.reshape(glob.shape[0], 3, 3).unsqueeze(1)
            ret = torch.matmul(glob, ret)
        # if we chose to have the identity as the result when the prediction is 0,
        if self.__IDENTITY_INIT:
            for i in range(0, 3):
                ret[:, :, i, i] += 1

        return ret

    def extract_code(self, source, target):
        '''
		given a batch, extract the global code w.r.t the source and targets, using the set encoders
		:param batch: the batch object
		:return: Bx|z| batch of codes z
		'''
        return self.encoder.encode_deformation(source, target)

    def hardpoisson(self, source, jacobians, vertices, faces, cut_values, vertex_pairs, threshold=0.5):
        """ cut_values: E*2 x 1 array of values to evaluate against threshold to determine cuts"""
        assert len(vertex_pairs)  == len(cut_values), f"vertex pairs len: {len(vertex_pairs)}, cut_values len: {len(cut_values)}"
        # Compute new topology based off the weights (under threshold => have edge/no cut)
        seamlengths = []
        mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())

        ## Cutting strategy
        # Map weights => edge cuts
        # Construct edge cut list by connecting cut paths
        # Loop through edges and cut with prio:
        # (1) edge on boundary
        # (2) two edges which share a vertex
        # (3) worst case: have to cut internal edge while choosing random adjacent edge as e2_i
        from meshing.edit import EdgeCut

        # NOTE: If cut values are vertex separation, then either vertex split can result in edge being cut
        # Weights correspond to vertex pairs which refer to edges (edgeidxs)
        cut_e = np.unique(source.edges[np.where(cut_values > threshold)[0]]) # Dupes b/c vertex pair:edges is 2:1

        # Filter out bad edges (edges on boundary)
        bd_edges = []
        for candidate_e in cut_e:
            if mesh.topology.edges[candidate_e].onBoundary():
                bd_edges.append(candidate_e)
        if len(bd_edges) > 0:
            print(f"Hard Poisson: Found {len(bd_edges)} boundary edges in cut edges. Deleting ... ")
            cut_e = np.delete(cut_e, np.where(np.isin(cut_e, bd_edges)))

        ### Construct edge cut list (recursive tree search) & make cuts
        edge_list, vertex_cutlist, seamlengths = order_edges_and_cut(mesh, cut_e)

        # List of cutlists (edge indices)

        # Re-sort so that cutlists which start at boundary are first
        # Make sure that subsequent cutlists are prio if they start on boundaries
        # NOTE: OG cutlists will be edge indices bc the indexing doesn't change after each cut -- BUT the vertex indices will change!

        # Store indices of vertices which correspond to cut edges
        # TODO: Cut edges indexing is incorrect! Need to update for subsequent cuts.
        hp_cutedges = []
        while len(cut_e) > 0:
            eidx = None
            vidx = None
            # (1) One vertex on boundary
            for candidate_e in cut_e:
                if np.any([v.onBoundary() for v in mesh.topology.edges[candidate_e].two_vertices()]):
                    eidx = candidate_e
                    edge = mesh.topology.edges[candidate_e]
                    vidx = edge.halfedge.vertex.index if edge.halfedge.vertex.onBoundary() else edge.halfedge.twin.vertex.index
                    break
            if eidx is not None and vidx is not None:
                cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
                EdgeCut(mesh, eidx, vidx, cutbdry=True, checktopo=False).apply()
                cut_e = np.delete(cut_e, np.where(cut_e == eidx))
                continue

            # (2) Two edges adjacent to each other
            eidx = None
            eidx2 = None
            for candidate_e in cut_e:
                for candidate_e2 in mesh.topology.edges[candidate_e].halfedge.vertex.adjacentEdges():
                    if candidate_e2.index in cut_e and candidate_e2.index != candidate_e:
                        eidx = candidate_e
                        eidx2 = candidate_e2.index
                        break

                if eidx is not None and eidx2 is not None:
                    break

                for candidate_e2 in mesh.topology.edges[candidate_e].halfedge.twin.vertex.adjacentEdges():
                    if candidate_e2.index in cut_e and candidate_e2.index != candidate_e:
                        eidx = candidate_e
                        eidx2 = candidate_e2.index
                        break

                if eidx is not None and eidx2 is not None:
                    break

            if eidx is not None and eidx2 is not None:
                cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                cute1 = [v.index for v in mesh.topology.edges[eidx2].two_vertices()]
                cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
                hp_cutedges.append(cute1)
                hp_cutedges.append(cute2)

                seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
                seamlengths.append(mesh.length(mesh.topology.edges[eidx2]))

                vidx = list(set(mesh.topology.edges[eidx].two_vertices()).intersection(mesh.topology.edges[eidx2].two_vertices()))[0].index
                EdgeCut(mesh, eidx, vidx, cutbdry=True, e2_i=eidx2, checktopo=False).apply()
                cut_e = np.delete(cut_e, np.where(np.isin(cut_e, [eidx, eidx2])))
                continue

            # (3) Worst case: choose random adjacent second edge that's not on boundary and not going to create a disconnected component
            # TODO: CHECK FOR POSSIBLE DISCONNECTION
            eidx = cut_e[0]
            for candidate_e2 in mesh.topology.edges[eidx].halfedge.vertex.adjacentEdges():
                if candidate_e2.index != eidx and not candidate_e2.onBoundary():
                    eidx2 = candidate_e2.index
                    break

            if eidx is not None or eidx2 is not None:
                for candidate_e2 in mesh.topology.edges[eidx].halfedge.twin.vertex.adjacentEdges():
                    if candidate_e2.index != eidx and not candidate_e2.onBoundary():
                        eidx2 = candidate_e2.index
                        break

            if eidx is None or eidx2 is None:
                raise ValueError(f"Hard Poisson: no valid cuts available for edge cut list {cut_e}.")

            cute1 = [v.index for v in mesh.topology.edges[eidx].two_vertices()]
            cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
            hp_cutedges.append(cute1)
            hp_cutedges.append(cute2)

            cute1 = [v.index for v in mesh.topology.edges[eidx2].two_vertices()]
            cute2 = [cute1[0], len(mesh.topology.vertices)] if vidx != cute1[0] else [cute1[1], len(mesh.topology.vertices)]
            hp_cutedges.append(cute1)
            hp_cutedges.append(cute2)

            seamlengths.append(mesh.length(mesh.topology.edges[eidx]))
            seamlengths.append(mesh.length(mesh.topology.edges[eidx2]))

            vidx = list(set(mesh.topology.edges[eidx].two_vertices()).intersection(mesh.topology.edges[eidx2].two_vertices()))[0].index

            EdgeCut(mesh, eidx, vidx, cutbdry=True, e2_i=eidx2, checktopo=False).apply()
            cut_e = np.delete(cut_e, np.where(np.isin(cut_e, [eidx, eidx2])))

        seamlengths = np.array(seamlengths)
        cutvs, cutfs, cutes = mesh.export_soup()
        meshprocessor = MeshProcessor.MeshProcessor.meshprocessor_from_array(cutvs, cutfs, source.source_dir, source._SourceMesh__ttype,
                                                                             cpuonly=source.cpuonly, load_wks_samples=source._SourceMesh__use_wks,
                                                                             load_wks_centroids=source._SourceMesh__use_wks,
                                                                             sparse=self.args.sparsepoisson)
        meshprocessor.prepare_temporary_differential_operators(source._SourceMesh__ttype)
        poissonsolver = meshprocessor.diff_ops.poisson_solver
        poissonsolver.to(self.device)

        harduv = poissonsolver.solve_poisson(jacobians).squeeze()

        return harduv, cutvs, cutfs, seamlengths, hp_cutedges

    #######################################
    # Pytorch Lightning Boilerplate code (training, logging, etc.)
    #######################################
    def training_step(self, source_batches, batch_id):
        # start = time.time()
        # torch.cuda.synchronize()

        # if self.args.debug:
        #     import pdb
        #     pdb.set_trace()

        if self.args.debug:
            import cProfile, pstats, sys

            pr = cProfile.Profile()
            pr.enable()

            batch_parts = self.my_step(source_batches, batch_id)

            pr.disable()
            ps = pstats.Stats(pr, stream=sys.stdout)
            ps.sort_stats('cumulative')
            ps.print_stats(.01)
        else:
            batch_parts = self.my_step(source_batches, batch_id)

        # Log scheduled weights
        if self.args.stitchschedule:
            self.log('edgecut_weight', self.args.edgecut_weight, logger=True, batch_size=1)
        if self.args.sparse_schedule:
            self.log('sparsecuts_weight', self.args.sparsecuts_weight, logger=True, batch_size=1)

        loss = batch_parts["loss"]
        lossrecord = batch_parts["lossdict"]

        # if self.args.debug:
        #     import pdb
        #     pdb.set_trace()

        self.log("train_loss", loss, logger=True, prog_bar=True, batch_size=1, on_epoch=True, on_step=False)

        # Log losses
        for key, val in lossrecord[0].items():
            if "loss" in key:
                self.log(key, np.mean(val), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        if self.current_epoch % self.args.val_interval == 0:
            if self.args.debug:
                import cProfile, pstats, sys

                pr = cProfile.Profile()
                pr.enable()

                self.plot_results(source_batches, batch_parts, batch_id, validation=False)

                pr.disable()
                ps = pstats.Stats(pr, stream=sys.stdout)
                ps.sort_stats('cumulative')
                ps.print_stats(.01)
            else:
                self.plot_results(source_batches, batch_parts, batch_id, validation=False)

        if self.args.mem:
            # Check memory consumption
            # Get GPU memory usage
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            m = torch.cuda.max_memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.")

            # Get CPU RAM usage too
            import psutil
            print(f'RAM memory % used: {psutil.virtual_memory()[2]}')

        return loss

    def on_train_epoch_end(self):
        self.log("epoch", self.current_epoch)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=False, logger=True, batch_size=1)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def get_gt_map(self, source, target, softpoisson=False):
        GT_V = target.get_vertices()

        # ground truth jacobians, restricted as well
        GT_J = source.jacobians_from_vertices(GT_V)
        GT_J_restricted = source.restrict_jacobians(GT_J)
        return GT_V, GT_J, GT_J_restricted

    def predict_uv(self, source, target, inituv=None):
        stacked = source.get_input_features()
        vertexcodes = self.forward(stacked, source)

        if inituv is not None:
            if self.args.inituvinput:

                if self.args.fftuv > 0:
                    inituv = self.fftuv(inituv)

                # NOTE: boundary weights will default to 0
                vertexcodes = torch.cat([vertexcodes, inituv], dim=1)

        pred_uvs = self.uv_mlp(vertexcodes).squeeze() # V x 2

        # Delta against initial UVs (must be fuv format)
        if inituv is not None:
            pred_uvs = pred_uvs + inituv

        # Convert to fuvs
        pred_uvs = pred_uvs[source.get_source_triangles()]

        return pred_uvs

    def predict_edge(self, source, target):
        stacked = source.get_input_features()

        # Convert vertex features to symmetric edge features
        edge_vpairs_nobound = source.ogedge_vpairs_nobound.detach().cpu().numpy()

        # Vertex features => MeshCNN edge features
        meshcnn_edge_vpairs = source.edges
        edgecodes = torch.cat([torch.abs(stacked[meshcnn_edge_vpairs[:,0]] - stacked[meshcnn_edge_vpairs[:,1]]),
                                stacked[meshcnn_edge_vpairs[:,0]] + stacked[meshcnn_edge_vpairs[:,1]]], dim=1)

        # NJF edges => MeshCNN edges
        edge2key = source.edge2key
        meshcnn_to_njfnobound = []
        for i, edge in enumerate(edge_vpairs_nobound):
            meshcnn_to_njfnobound.append(edge2key[tuple(sorted(edge))])
        meshcnn_to_njfnobound = np.array(meshcnn_to_njfnobound)

        if self.args.initweightinput:
            initweights = source.initweights.to(self.device).unsqueeze(1)

            if self.args.fftweight > 0:
                initweights = self.fftweight(initweights)

            # NOTE: boundary weights will default to 0
            mapped_weights = torch.zeros(len(edgecodes), 1, dtype=initweights.dtype).to(self.device)
            mapped_weights[meshcnn_to_njfnobound] = initweights
            edgecodes = torch.cat([edgecodes, mapped_weights], dim=1)

        edgecodes = edgecodes.transpose(1, 0) # features x E
        edgevals = self.edge_decoder(edgecodes.unsqueeze(0), [source]).squeeze().transpose(1,0)
        edgevals = self.edgeweight_mlp(edgevals).squeeze()

        # MeshCNN edges => NJF edges
        edgevals = edgevals[meshcnn_to_njfnobound]

        return edgevals

    def predict_map(self, source, target, initj=None):
        if self.args.softpoisson or self.args.optweight:
            pred_J, weights = self.predict_jacobians(source, target, initj)
        else:
            pred_J = self.predict_jacobians(source, target, initj)

        # Need initialization J to have batch dimension
        if initj is not None:
            if len(initj.shape) == 3:
                initj = initj.unsqueeze(0)

            pred_J = torch.cat([torch.einsum("abcd,abde->abce", (pred_J[:,:,:2,:2], initj[:,:,:2,:])),
                                torch.zeros(pred_J.shape[0], pred_J.shape[1], 1, pred_J.shape[3], device=self.device)],
                               dim=2) # B x F x 3 x 3

        if self.args.align_2D:  # if the target is in 2D the last column of J is 0
            pred_J[:, :,2,:] = 0 # NOTE: predJ shape is B x F x 3 x 3 (where first 3-size is interpreted as axes)

        # Update the soft poisson if set
        updatedlap = False
        if self.args.softpoisson or self.args.optweight:
            # Reorganize predicted weights to match soft poisson indexing
            sp_idxs = source.edge_vpairs.reshape(-1, 2) # E * 2 x 2 (last dimension compares corresponding vertices (two pairs per edge))
            sp_idxs = (torch.cat([sp_idxs[:,0],sp_idxs[:,1]]), torch.cat([sp_idxs[:,1],sp_idxs[:,0]]))

            sp_weights = weights.repeat_interleave(2)
            sp_weights = torch.cat([sp_weights, sp_weights])

            source.poisson.lap.update_vals(sp_weights, sp_idxs, source.poisson.lap_pinned,
                                            source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)

            # if self.args.softpoisson == "edges":
            #     source.poisson.lap.update_vals(weights, source.valid_edge_pairs, source.poisson.lap_pinned,
            #                                 source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)
            # elif self.args.softpoisson == "valid":
            #     source.poisson.lap.update_vals(weights, source.valid_pairs, source.poisson.lap_pinned,
            #                                 source.poisson.lap_pinned_rows, source.poisson.lap_pinned_cols)

            updatedlap = True

        pred_V = source.vertices_from_jacobians(pred_J, updatedlap = updatedlap)

        # Get back jacobians from predicted vertices
        # TODO: For now we use pred_J for the predicted jacobians, but we could also use the restriction.
        # My intuition is that using pred_J while spoofing restriction (by dropping the 3rd row) is better for NN since basis
        # restriction can result in pretty arbitrary transformation across neighboring triangles but TBD.

        pred_J_poiss = source.poisson.jacobians_from_vertices(pred_V)
        pred_J_restricted_poiss = source.restrict_jacobians(pred_J)

        if self.args.softpoisson or self.args.optweight:
            return pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss, -weights

        return pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss

    def check_map(self, source, target, GT_J, GT_V):
        pred_V = source.vertices_from_jacobians(GT_J)
        return torch.max(torch.absolute(pred_V - GT_V))


    def validation_step(self, batch, batch_idx):

        if self.args.debug:
            import cProfile, pstats, sys

            pr = cProfile.Profile()
            pr.enable()

            batch_parts = self.my_step(batch, batch_idx, validation=True)

            pr.disable()
            ps = pstats.Stats(pr, stream=sys.stdout)
            ps.sort_stats('cumulative')
            ps.print_stats(.01)
        else:
            batch_parts = self.my_step(batch, batch_idx, validation=True)

        val_loss = None
        if 'loss' in batch_parts:
            val_loss = batch_parts['loss'].item()
            self.log('val_loss', val_loss, logger=True, prog_bar=True, batch_size=1, on_epoch=True, on_step=False)

        self.val_step_iter += 1

        if torch.rand(1).item() > self.args.valrenderratio:
            return val_loss

        if self.args.debug:
            import cProfile, pstats, sys

            pr = cProfile.Profile()
            pr.enable()

            self.plot_results(batch, batch_parts, batch_idx, validation=True)

            pr.disable()
            ps = pstats.Stats(pr, stream=sys.stdout)
            ps.sort_stats('cumulative')
            ps.print_stats(.01)
        else:
            self.plot_results(batch, batch_parts, batch_idx, validation=True)

        return val_loss

    ### TRAINING STEP HERE ###
    def my_step(self, source_batch, batch_idx, validation=False):
        # sanity checking the poisson solve, getting back GT vertices from GT jacobians. This is not used in this training.
        # GTT = batches.get_batch(0).poisson.jacobians_from_vertices(pred_V[0])
        # 		GT_V = batches.get_batch(0).poisson.solve_poisson(GTT)
        source = source_batch[0]
        target = source_batch[1]
        source.to(self.device)
        target.to(self.device)

        if not hasattr(self.args, "gtedgeloss"):
            self.args.gtedgeloss = None

        # Initialize stitchweights if not yet set
        if batch_idx not in self.stitchweights:
            self.stitchweights[batch_idx] = torch.ones(len(source.valid_pairs), device=self.device)

        initj = None
        inituv = None
        sourcedim = 3
        if self.args.init == "tutte":
            sourcedim = 2
            initj = source.tuttej.squeeze().to(self.device)
            inituv = source.tutteuv.squeeze().to(self.device)
        elif self.args.init == "isometric":
            sourcedim = 2
            initj = source.isoj.squeeze().to(self.device) # NOTE: this is guaranteed to be isometric, so don't need to composite for computing distortion
            inituv = source.isouv.squeeze().to(self.device)
        elif self.args.init == "slim":
            sourcedim = 2
            initj = source.slimj.squeeze().to(self.device)
            inituv = source.slimuv.squeeze().to(self.device)
        elif self.args.init == "precut":
            sourcedim = 2
            initj = source.prej.squeeze().to(self.device)
            inituv = source.preuv.squeeze().to(self.device)

        # Need to export mesh soup to get correct face to tutte uv indexing
        mesh = Mesh(source.get_vertices().detach().cpu().numpy(), source.get_source_triangles())
        vs, fs, es = mesh.export_soup()
        vertices = torch.from_numpy(vs).float().to(self.device)
        faces = torch.from_numpy(fs).long().to(self.device)

        pred_V = pred_J = pred_J_poiss = pred_J_restricted_poiss = weights = None
        if self.arch == "directuv":
            preduv = self.predict_uv(source, target, inituv = inituv[:,:2])
        elif self.arch == "directedge":
            predcuts = self.predict_edge(source, target)
        else:
            if self.args.softpoisson or self.args.optweight:
                pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss, weights = self.predict_map(source, target, initj=initj if initj is not None else None)
            else:
                pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss = self.predict_map(source, target, initj=initj if initj is not None else None)

            # Drop last dimension of restricted J
            if pred_J_restricted_poiss.shape[2] == 3:
                pred_J_restricted_poiss = pred_J_restricted_poiss[:,:,:2]

        ## Stitching loss schedule
        if self.args.stitchschedule == "linear":
            ratio = self.current_epoch/self.trainer.max_epochs
            edgecut_weight = ratio * self.args.edgecut_weight_max + (1 - ratio) * self.args.edgecut_weight_min
            self.args.edgecut_weight = edgecut_weight
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight
        elif self.args.stitchschedule == "cosine":
            ratio = self.current_epoch/self.args.sparse_cosine_steps
            edgecut_weight = self.args.edgecut_weight_max - 0.5 * (self.args.edgecut_weight_max - self.args.edgecut_weight_min) * (1 + np.cos(np.pi * ratio))
            self.args.edgecut_weight = edgecut_weight
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight
        elif self.args.stitchschedule == "constant":
            ratio = self.current_epoch/self.trainer.max_epochs
            if ratio < self.args.stitchschedule_constant:
                edgecut_weight = 0
                self.args.edgecut_weight = 0
            else:
                edgecut_weight = self.args.edgecut_weight_max
                self.args.edgecut_weight = self.args.edgecut_weight_max
            # NOTE: Need this hack bc when lossfcn is loaded from weights args is a separate object
            self.lossfcn.args.edgecut_weight = edgecut_weight

        ## Sparse loss schedule
        if self.args.sparse_schedule == "linear":
            ratio = self.current_epoch/self.trainer.max_epochs
            sparsecuts_weight = ratio * self.args.sparselossweight_max + (1 - ratio) * self.args.sparselossweight_min
            self.args.sparsecuts_weight = sparsecuts_weight
        elif self.args.sparse_schedule == "cosine":
            ratio = self.current_epoch/self.args.sparse_cosine_steps
            sparsecuts_weight = self.args.stitchlossweight_max - 0.5 * (self.args.stitchlossweight_max - self.args.stitchlossweight_min) * (1 + np.cos(np.pi * ratio))
            self.args.sparsecuts_weight = sparsecuts_weight

        normalgrid = None
        if self.args.normalloss:
            # NOTE: assumes batch size is always 1
            normalgrid = source.get_loaded_data('normalgrid')

        # NOTE predict_map already composites pred_J against initj
        lossrecord = [{}]
        if not self.args.novelinference:
            if self.arch == "directuv":
                loss = self.lossfcn.computeloss(vertices, faces, preduv.reshape(-1, 3, 2),
                                                weights=weights, stitchweights=self.stitchweights[batch_idx],
                                                source=source, keepidxs=source.keepidxs, mesh = mesh,
                                                normalgrid=normalgrid)
            elif self.arch == "directedge":
                loss = self.lossfcn.computeloss(vertices, faces,weights=predcuts, source=source,)
            else:
                pred_V = pred_V[:, :, :2].squeeze().reshape(-1, 3, 2)
                loss = self.lossfcn.computeloss(vertices, faces, ZeroNanGrad.apply(pred_V), ZeroNanGrad.apply(pred_J_poiss[:,:,:2,:]),
                                                weights=weights, stitchweights=self.stitchweights[batch_idx],
                                                source=source, keepidxs=source.keepidxs, mesh = mesh, predj = pred_J,
                                                normalgrid=normalgrid)

            lossrecord = self.lossfcn.exportloss()
            self.lossfcn.clear() # This resets the loss record dictionary

        ### ==== SDS Losses ==== ###
        if self.args.sdsloss:
            # Set cache directories
            # import os
            # os.environ['HF_DATASETS_CACHE'] = "/net/scratch/rliu/.cache"
            # os.environ['HF_HOME'] = "/net/scratch/rliu/.cache"

            # Prereqs: texture image, texture description
            assert self.args.textureimg is not None and self.args.texturetext is not None, "Need to specify textureimg and texturetext for SDS loss"
            from diffusion_guidance.deepfloyd_if import DeepFloydIF, DeepFloydIF_Img2Img

            if self.args.sdsloss == "text2img":
                diffusion = DeepFloydIF() # optionally you can pass a config at initialization
            elif self.args.sdsloss == "img2img":
                diffusion = DeepFloydIF_Img2Img()

            # Text encoding
            texturetext = ' '.join(self.args.texturetext)
            text_z, text_z_neg = diffusion.encode_prompt(texturetext)

            # TODO: Render texture image on mesh for randomly sampled views
            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor
            textureimg = pil_to_tensor(Image.open(self.args.textureimg)).double().to(self.device)

            rgb_images = []
            # NOTE: Can upscale resolution to get better gradients
            from renderer import render_texture

            num_views = 5
            radius = 2.5
            center = torch.zeros(2)
            azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
               :-1].double().to(self.device)
            elev = torch.zeros(len(azim), device=self.device).double()

            # Face UVs
            uv_face = pred_V # F x 3 x 2

            # Need to scale UVs between 0-1
            uv_face = uv_face - torch.min(uv_face.reshape(-1, 2), dim=0)[0]
            uv_face = uv_face/torch.max(uv_face)

            ## DEBUGGING
            # if self.args.debug:
            #     uv_face = torch.load("/net/scratch/rliu/NJFWand/data/cone/gt_uvs.pt").reshape(-1, 3, 2).to(self.device)

            #     # Need to scale UVs between 0-1
            #     uv_face -= torch.min(uv_face.reshape(-1, 2), dim=0)[0]
            #     uv_face /= torch.max(uv_face)

            rgb_images.append(render_texture(vertices.double(), faces, uv_face, elev, azim, radius, textureimg/255, lights=None,
                                                  resolution=(128, 128), device=self.device, lookatheight=0, whitebg=True,
                                                  interpolation_mode='bilinear'))

            sds = diffusion(rgb_images[0]['image'], text_z)
            sdsloss = sds['loss_sds']

            loss = loss + sdsloss
            lossrecord[0]['sdsloss'] = sdsloss.cpu().detach().numpy()
            lossrecord[0]['target'] = sds['target'].cpu().detach().numpy()

        if self.args.imageloss:
            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor
            from torchvision.transforms import Resize
            import torchvision
            from source_njf.utils import normalize_uv

            vizuv = pred_V.clone()
            with torch.no_grad():
                normalize_uv(vizuv)
                # Scale back to centered at 0.5
                vizuv += 0.5

            texturename = os.path.basename(self.args.textureimg).split(".")[0]
            img = Image.open(self.args.textureimg)
            img = img.convert("RGB")
            textureimg = pil_to_tensor(img).double().to(self.device)

            rgb_images = []

            from source_njf.renderer import render_texture
            total_views = 5
            radius = 2.5
            center = torch.zeros(2)
            azim = torch.linspace(center[0], 2 * np.pi + center[0], total_views + 1)[
                :-1].double().to(self.device)
            elev = torch.zeros(len(azim), device=self.device).double()

            pred_images = render_texture(vertices.double(), faces, vizuv, elev, azim, radius, textureimg/255, lights=None,
                                                    resolution=(self.args.renderresolution, self.args.renderresolution), device=self.device, lookatheight=0, whitebg=True,
                                                    interpolation_mode = self.args.interpmode)
            pred_images = Resize((self.args.resolution, self.args.resolution))(pred_images)
            rgb_images.append(pred_images)

            gt_images = []
            for i in range(total_views):
                gt_image = torchvision.io.read_image(self.args.imageloss + f"_{i}.png").double().to(self.device)
                gt_image = Resize((self.args.resolution, self.args.resolution))(gt_image)/255
                gt_images.append(gt_image)

            gt_images = torch.stack(gt_images, dim=0)
            imageloss = torch.nn.functional.mse_loss(rgb_images[0]['image'], gt_images, reduction="none")
            loss += torch.mean(imageloss)
            lossrecord[0]['imageloss'].append(imageloss.cpu().detach().numpy())

        # If running stitchweights, then update here
        # NOTE: stitchweights is len(valid_pairs) but weights is len(valid_edges)
        if self.args.stitchweight and self.args.softpoisson:
            if self.args.stitchweight == "stitchloss":
                self.stitchweights[batch_idx] = 1/(lossrecord[0]['edgeseparation'] + 1e-8)
                assert self.stitchweights[batch_idx].requires_grad == False
            elif self.args.stitchweight == "softweight":
                if self.args.softpoisson == "edges":
                    self.stitchweights[batch_idx][source.edgeidxs] = weights
                else:
                    self.stitchweights[batch_idx] = weights
            elif self.args.stitchweight == "softweightdetach":
                if self.args.softpoisson == 'edges':
                    self.stitchweights[batch_idx][source.edgeidxs] = weights.detach()
                else:
                    self.stitchweights[batch_idx] = weights.detach()
                assert self.stitchweights[batch_idx].requires_grad == False
            else:
                raise ValueError(f"Unknown stitchweight {self.args.stitchweight}")

        if self.verbose:
            print(
                f"batch of {target.get_vertices().shape[0]:d} source <--> target pairs, each mesh {target.get_vertices().shape[1]:d} vertices, {source.get_source_triangles().shape[1]:d} faces")

        if self.arch == "directuv":
            ret = {
                "target_V": vertices.detach(),
                "source_V": vertices.detach(),
                "pred_V": preduv.reshape(-1, 2).detach(),
                'T': np.arange(len(faces) * 3).reshape(len(faces), 3),
                "ogT": faces.detach().cpu().numpy(),
                'source_ind': source.source_ind,
                'target_inds': target.target_inds,
                "lossdict": lossrecord,
            }

            # Get jacobians
            from source_njf.utils import get_jacobian_torch
            fverts = vertices[faces].reshape(-1, 3)
            ret['predJ'] = get_jacobian_torch(fverts, torch.arange(len(fverts)).reshape(-1, 3).to(fverts.device), preduv.reshape(-1, 2),
                                              device=self.device)

            return ret

        elif self.arch == "directedge":
            ret = {
                "target_V": vertices.detach(),
                "source_V": vertices.detach(),
                "weights": predcuts.detach().cpu().numpy(),
                'T': np.arange(len(faces) * 3).reshape(len(faces), 3),
                "ogT": faces.detach().cpu().numpy(),
                'source_ind': source.source_ind,
                'target_inds': target.target_inds,
                "lossdict": lossrecord,
            }

        if not self.args.novelinference:
            ret['loss'] = loss

        return ret

    def colors(self, v, f):
        vv = igl.per_vertex_normals(v, f)
        vv = (numpy.abs(vv) + 1) / 2
        colors = vv * 255
        return torch.from_numpy(colors).unsqueeze(0)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        # if self.global_step % FREQUENCY == 0:
        #     # hack to ensure cupy doesn't leak on us
        #     mempool = cupy.get_default_memory_pool()
        #     pinned_mempool = cupy.get_default_pinned_memory_pool()
        #     mempool.free_all_blocks()
        #     pinned_mempool.free_all_blocks()
        return

    def get_layer_normalization_type(self):
        if hasattr(self.args, 'layer_normalization'):
            layer_normalization = self.args.layer_normalization
        else:
            assert hasattr(self.args, 'batchnorm_decoder')
            layer_normalization = self.args.batchnorm_decoder
        return layer_normalization

    def get_pointnet_layer_normalization_type(self):
        if hasattr(self.args, 'pointnet_layer_normalization'):
            layer_normalization = self.args.pointnet_layer_normalization
        else:
            assert hasattr(self.args, 'batchnorm_encoder')
            layer_normalization = self.args.batchnorm_encoder
        return layer_normalization

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        pass

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(list(self.parameters()), lr=self.lr)
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(list(self.parameters()), lr=self.lr)

        # TODO: Consider having separate LR between face latents vs jacobians
        # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.lr_epoch_step[0],self.args.lr_epoch_step[1]], gamma=0.1)
        # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8, threshold=0.0001,
        #                                                         min_lr=1e-7, verbose=True)

        # Add weight as additional parameter
        if self.args.optweight:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            for i, bundle in enumerate(dataloader):
                source, target = bundle
                initweights = source.initweights
                optweights = torch.zeros(initweights.shape).to(self.device).double() + 1e-7
                optweights.requires_glrad_()
                additional_parameters = [optweights]
                optimizer.add_param_group({"params": additional_parameters, 'lr': self.lr})

        # Add texture optimization as additional parameter (from NUVO)
        # Resolution: 256 x 256
        if self.args.normalloss:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            for i, bundle in enumerate(dataloader):
                source, target = bundle
                optimizer.add_param_group({"params": [source.get_loaded_data('normalgrid')], 'lr': self.lr})

        return {"optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler1,
                #     "monitor": "train_loss",
                #     },
                }

    def plot_results(self, batch, batch_parts, batch_idx, validation=False):
        ### Visualizations
        import matplotlib.pyplot as plt

        # Log path
        pref = 'train'
        if validation:
            pref = 'val'

        source, target = batch
        sourcename = os.path.basename(source.source_dir)
        source_path = os.path.join(self.logger.save_dir, "renders", sourcename)
        save_path = os.path.join(source_path, "frames")
        pred_path = os.path.join(self.logger.save_dir, "predictions", pref, sourcename)

        # Loss dict
        lossdict = batch_parts['lossdict']
        keepidxs = source.keepidxs

        # Construct mesh
        mesh = Mesh(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"])

        # 3d positions as uv colors
        fvs = mesh.vertices[mesh.faces] # F x 3 x 3
        fpos = np.mean(fvs, axis=1)
        fpos -= np.min(fpos, axis=0, keepdims=True)
        fpos /= np.max(np.linalg.norm(fpos, axis=1))

        # Log losses
        if validation:
            for key, val in lossdict[0].items():
                if "loss" in key:
                    self.log(f"val_{key}", np.mean(val), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        if not os.path.exists(save_path):
            Path(save_path).mkdir(exist_ok=True, parents=True)

        if not os.path.exists(pred_path):
            Path(pred_path).mkdir(exist_ok=True, parents=True)

        # Save latest predictions
        if 'pred_V' in batch_parts.keys():
            np.save(os.path.join(pred_path, f"latest_preduv.npy"), batch_parts['pred_V'].squeeze().detach().cpu().numpy())

        if 'T' in batch_parts.keys():
            np.save(os.path.join(pred_path, f"latest_predt.npy"), batch_parts['T'])

        if self.args.softpoisson:
            np.save(os.path.join(pred_path, f"latest_predw.npy"), batch_parts['weights'])
            np.save(os.path.join(pred_path, f"latest_predj.npy"), batch_parts["pred_J"].squeeze().detach().cpu().numpy())

        # Plot the histogram of weights
        if self.args.softpoisson or self.args.arch == "directedge":
            fig, axs = plt.subplots()
            # plot ours
            axs.set_title(f"{pref} Epoch {self.current_epoch:05}: SP Weights")
            axs.hist(batch_parts['weights'], bins=20)
            plt.savefig(os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png"))
            plt.close(fig)
            plt.cla()
            self.logger.log_image(key=f'{pref} weights', images=[os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png")],
                                    step=self.current_epoch)

        #### If directedge arch: choose edges to cut and export the cut mesh
        if self.arch == "directedge" and self.args.test:
            from igl import cut_mesh

            weights =  batch_parts['weights']
            cut_edges = np.where(weights < 0.5)[0]
            cut_vpairs = source.ogedge_vpairs_nobound[cut_edges].detach().cpu().numpy().tolist()

            cutfes = np.zeros((len(mesh.faces), 3)).astype(int)
            edges_list = []
            for f in mesh.faces:
                for i in range(3):
                    e = [f[i], f[(i+1)%3]]
                    edges_list.append(e)

            for i in range(len(edges_list)):
                edgecheck = edges_list[i]
                if edgecheck in cut_vpairs or edgecheck[::-1] in cut_vpairs:
                    cutfes[i // 3, i % 3] = 1

            cutvs, cutfs = cut_mesh(mesh.vertices, mesh.faces, cutfes)
            cutmesh = Mesh(cutvs, cutfs)
            cutmesh.export_obj(pred_path, f"{source.source_ind}_edgecut")

        # Texture images
        # Save rendered images for debugging
        if self.args.sdsloss:
            import matplotlib.pyplot as plt

            images = batch_parts['textureimgs']
            num_views = 5
            fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
            for nview in range(num_views):
                j = nview % 5
                if nview > 5:
                    i = nview // 5
                    axs[i,j].imshow(images[nview].transpose(1,2,0))
                    axs[i,j].axis('off')
                else:
                    axs[j].imshow(images[nview].transpose(1,2,0))
                    axs[j].axis('off')
            plt.axis('off')
            fig.suptitle(f"{pref} Epoch {self.current_epoch} Textures")
            plt.savefig(os.path.join(save_path, f"{self.current_epoch:05}_{source.source_ind}_texture.png"))
            plt.close(fig)
            plt.cla()

            # Log the plotted imgs
            images = [os.path.join(save_path, f"{self.current_epoch:05}_{source.source_ind}_texture.png")]
            self.logger.log_image(key=f'{pref} textures', images=images, step=self.current_epoch)

        # NOTE: batch_parts['T'] = triangle soup indexing if no poisson solve
        # If recutting Tutte: then plot the original tutte uvs
        if (self.args.init in ["tutte", 'slim', 'isometric', 'precut'] and self.args.ninit == -1) or \
            (self.current_epoch == 0 and self.args.init):
            source = batch[0]
            if self.args.init == "tutte":
                uv = source.tutteuv
                uvfs = source.cutfs
            elif self.args.init == "slim":
                uv = source.slimuv
                uvfs = source.cutfs
            elif self.args.init == "isometric":
                uv = source.isofuv.reshape(-1, 2)
                uvfs = np.arange(len(uv)).reshape(-1, 3)
            elif self.args.init == "precut":
                uv = source.preuv.squeeze()
                uvfs = np.arange(len(uv)).reshape(-1, 3)
            else:
                raise ValueError(f"Unknown init type: {self.args.init}")

            plot_uv(save_path, f"{pref} {self.args.init} init epoch {self.current_epoch:05} batch {batch_idx}", uv.squeeze().detach().cpu().numpy(),
                    uvfs, losses=None, facecolors = fpos,
                    stitchweight=self.args.edgecut_weight)

            # Also plot the full boundary
            initfaces = batch_parts["ogT"]
            cutmesh = Mesh(source.cutvs, source.cutfs)
            totboundaries = []
            for key, bd in sorted(cutmesh.topology.boundaries.items()):
                boundaryvs = []
                for v in bd.adjacentVertices():
                    boundaryvs.append(cutmesh.vertices[v.index])
                bidx = np.array([[i, i+1] for i in range(len(boundaryvs)-1)] + [[len(boundaryvs)-1, 0]])
                totboundaries.append(np.array(boundaryvs)[bidx])
            if len(totboundaries) > 0:
                totboundaries = np.concatenate(totboundaries, axis=0)
                export_views(source.cutvs, source.cutfs, save_path, filename=f"boundary_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"Initial Mesh Boundary Batch {batch_idx}", cylinders=totboundaries,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

        ### Compute cut edges based on stitching loss (on original topology and soup topology) ###
        if 'pred_V' in batch_parts.keys():
            if 'edgeseparation' not in lossdict[0].keys():
                edge_vpairs = source.edge_vpairs.detach().cpu().numpy()
                uvpairs = batch_parts["pred_V"].squeeze()[edge_vpairs, :] # E x 2 x 2 x 2
                uvpairs = uvpairs[keepidxs]

                edgeseparation = torch.sum(torch.nn.functional.l1_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2)
                edgeseparation = torch.mean(edgeseparation, dim=1).detach().cpu().numpy() # E x 1
            elif 'edgeseparation' in lossdict[0].keys():
                edgeseparation = lossdict[0]['edgeseparation']

            topo_cutedges_stitch = np.where(edgeseparation > self.args.cuteps)[0]
            topo_cutedgecolors_stitch = np.arange(len(edgeseparation))/(len(edgeseparation)-1)

            # Convert to soup edges
            # NOTE: We have to subset by keepidxs to get correct corr to edgeseparation
            soup_cutedges_stitch = []
            edge_vpairs = source.edge_vpairs.detach().cpu().numpy()[keepidxs]
            for cute in topo_cutedges_stitch:
                vpair = edge_vpairs[cute] # 2 x 2
                soup_cutedges_stitch.append(batch_parts["pred_V"].squeeze()[vpair[:,0]].detach().cpu().numpy())
                soup_cutedges_stitch.append(batch_parts["pred_V"].squeeze()[vpair[:,1]].detach().cpu().numpy())
            soup_cutedgecolors_stitch = np.repeat(topo_cutedges_stitch, 2)/(len(edgeseparation)-1)

            if len(soup_cutedges_stitch) > 0:
                soup_cutedges_stitch = np.stack(soup_cutedges_stitch, axis=0) # CutE * 2 x 2 x 2

        ### Compute cut edges based on weights (on original topology and soup topology) ###
        if "weights" in batch_parts.keys():
            weights = batch_parts['weights']
            # assert len(weights) == len(edgeseparation), f"weights len: {len(weights)}, edgeseparation len: {len(edgeseparation)}"

            topo_cutedges_weight = np.where(weights < self.args.weightcuteps)[0]
            topo_cutedgecolors_weight = np.arange(len(weights))/(len(weights)-1)

            # Convert to soup edges
            # soup_cutedges_weight = []
            # edge_vpairs = source.edge_vpairs.detach().cpu().numpy()
            # for cute in topo_cutedges_weight:
            #     vpair = edge_vpairs[cute] # 2 x 2
            #     soup_cutedges_weight.append(batch_parts["pred_V"].squeeze()[vpair[:,0]].detach().cpu().numpy())
            #     soup_cutedges_weight.append(batch_parts["pred_V"].squeeze()[vpair[:,1]].detach().cpu().numpy())
            # soup_cutedgecolors_weight = np.repeat(topo_cutedges_weight, 2)/(len(weights)-1)

            # if len(soup_cutedges_weight) > 0:
            #     soup_cutedges_weight = np.stack(soup_cutedges_weight, axis=0) # CutE * 2 x 2 x 2

        # Compute flips
        if 'pred_V' in batch_parts.keys():
            from utils import get_flipped_triangles
            flipped = get_flipped_triangles(batch_parts["pred_V"].squeeze().detach().cpu().numpy(), batch_parts['T'].squeeze())
            flipvals = np.zeros(len(batch_parts['T'].squeeze()))
            flipvals[flipped] = 1
            lossdict[0]['fliploss'] = flipvals

        # Compute distortion if not already in loss
        if 'pred_V' in batch_parts.keys():
            if 'distortionloss' not in lossdict[0].keys():
                from source_njf.losses import arap, symmetricdirichlet
                # distortionenergy = arap(batch_parts["source_V"], torch.from_numpy(batch_parts["ogT"]).to(self.device),
                #                         batch_parts["pred_V"],
                #                         paramtris = batch_parts["pred_V"].reshape(-1, 3, 2),
                #                         device=self.device,
                #                         renormalize=False,
                #                         return_face_energy=True, timeit=False)
                distortionenergy = symmetricdirichlet(batch_parts["source_V"], torch.from_numpy(batch_parts["ogT"]).to(self.device),
                                                    batch_parts['predJ'].squeeze())
                lossdict[0]['distortionloss'] = distortionenergy.detach().cpu().numpy()

                if validation:
                    self.log("val distortionloss", np.mean(distortionenergy.detach().cpu().numpy()), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

            if len(batch_parts["pred_V"].shape) == 4:
                for idx in range(len(batch_parts["pred_V"])):
                    plot_uv(save_path, f"{pref} epoch {self.current_epoch:05} batch {idx:05}", batch_parts["pred_V"][idx].squeeze().detach().cpu().numpy(),
                            batch_parts["T"][idx].squeeze(), losses=lossdict[idx], cmin=0, cmax=2, dmin=0, dmax=1,
                            facecolors = fpos,
                            edges = soup_cutedges_stitch, edgecolors = soup_cutedgecolors_stitch,
                            edgecorrespondences=source.edgecorrespondences, source=source,
                            keepidxs = keepidxs, stitchweight=self.args.edgecut_weight)
            else:
                plot_uv(save_path, f"{pref} epoch {self.current_epoch:05} batch {batch_idx}", batch_parts["pred_V"].squeeze().detach().cpu().numpy(),
                        batch_parts["T"].squeeze(), losses=lossdict[0], cmin=0, cmax=2, dmin=0, dmax=1,
                        facecolors = fpos,
                        edges = soup_cutedges_stitch, edgecolors = soup_cutedgecolors_stitch,
                        edgecorrespondences=source.edgecorrespondences, source=source,
                        keepidxs = keepidxs, stitchweight=self.args.edgecut_weight)

            # Log the plotted imgs
            images = [os.path.join(save_path, f"{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + \
                    [os.path.join(save_path, f"{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}_edges.png")] + \
                        [os.path.join(save_path, f"{key}_{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

            if self.args.init:
                images = [os.path.join(save_path, f"{pref}_{self.args.init}_init_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + images

            # Log GT UVs
            if self.args.gtuvloss or self.args.gtnetworkloss:
                images = [os.path.join(source.source_dir, "..", "..", "gtuv.png")] + images

            # Filter out all renders that dont exist
            images = [imgpath for imgpath in images if os.path.exists(imgpath)]

            # Make sure images are not locked before logging them
            from source_njf.utils import wait_for_file, is_locked
            import time

            count = 0
            while any([is_locked(imgpath) for imgpath in images]):
                if count > 10:
                    print("UV images locked for too long. Skipping logging.")
                    break
                time.sleep(1)
                count += 1

            if count <= 10:
                try:
                    self.logger.log_image(key=f'{pref} uvs', images=images, step=self.current_epoch)
                except Exception as e:
                    print(e)

            ### If we have fgroups, then plot the fgroups ###
            from matplotlib.tri import Triangulation
            import matplotlib.pyplot as plt

            if source.fgroups is not None:
                fgroups = source.fgroups
                preduv = batch_parts["pred_V"].squeeze().detach().cpu().numpy()
                sourcename = source.source_ind

                for fgroup in np.unique(fgroups):
                    fgroupmask = np.where(fgroups == fgroup)
                    groupuvs = preduv.reshape(-1, 3, 2)[fgroupmask].reshape(-1, 2)
                    triangles = np.arange(len(groupuvs)).reshape(-1, 3)
                    tris = Triangulation(groupuvs[:, 0], groupuvs[:, 1], triangles=triangles)
                    fig, axs = plt.subplots(figsize=(5, 5))
                    axs.set_title(f"{sourcename} Pred UVs (fgroup {fgroup})")
                    cmap = plt.get_cmap("tab20")
                    axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                        linewidth=0.1, edgecolor="black")
                    plt.axis('off')
                    axs.axis('equal')
                    plt.savefig(os.path.join(save_path, f"preduv_{fgroup}_{self.current_epoch:05}.png"))
                    plt.close(fig)
                    plt.cla()

                # Log
                images = []
                for fgroup in np.unique(fgroups):
                    images.append(os.path.join(source.source_dir, "..", "..", f"gtuv_{fgroup}.png"))
                    images.append(os.path.join(save_path, f"preduv_{fgroup}_{self.current_epoch:05}.png"))

                from source_njf.utils import wait_for_file, is_locked
                import time

                count = 0
                while any([is_locked(imgpath) for imgpath in images]):
                    if count > 10:
                        print("Weight images locked for too long. Skipping logging.")
                        break
                    time.sleep(1)
                    count += 1

                if count <= 10:
                    try:
                        self.logger.log_image(key=f'{pref} gt vs pred', images=images, step=self.current_epoch)
                    except Exception as e:
                        print(e)

        #### Compute and plot hard soft poisson ####
        # NOTE: We only look at valid edge pairs to cut, even if LEARNING is done over all valid pairs!
        # May also set weights based on the stitching loss instead
        # if self.args.hardpoisson:
        #     if self.args.hardpoisson == "loss":
        #         # NOTE: Edge stitch check is L2 distances between corresponding vertices!
        #         edgeweights = edgeseparation
        #         threshold = self.args.cuteps
        #     elif self.args.hardpoisson == "weight":
        #         edgeweights = weights
        #         threshold = self.args.weightcuteps

        #     hardpoisson_uv, cutvs, cutfs, seamlengths, hardpoisson_cuts = self.hardpoisson(source, batch_parts['pred_J'], batch_parts['source_V'],
        #                                                                     torch.from_numpy(batch_parts['ogT']).to(self.device), edgeweights,
        #                                                                     vertex_pairs = source.valid_edge_pairs,
        #                                                                     threshold = threshold)
        #     hardpoisson_uv = hardpoisson_uv[:, :2]

        #     # Compute seam length and distortion
        #     # NOTE: We only have ARAP implemented for now
        #     from source_njf.losses import arap
        #     distortionenergy = arap(torch.from_numpy(cutvs).float(), torch.from_numpy(cutfs).long(), hardpoisson_uv,
        #                             device=self.device, renormalize=False,
        #                             return_face_energy=True, timeit=False)

        #     # Visualize the edge cuts
        #     hp_ecolors, hp_edges = None, None
        #     hardpoisson_uv = hardpoisson_uv.squeeze().detach().cpu().numpy()

        #     plot_uv(save_path, f"{pref} hard poisson epoch {self.current_epoch:05} seam length {np.sum(seamlengths):04f}", hardpoisson_uv,
        #                 cutfs, losses={'distortionloss': distortionenergy.detach().cpu().numpy()},
        #                 edges = hp_edges, edgecolors=hp_ecolors, keepidxs = keepidxs,
        #                 facecolors = fpos, stitchweight=self.args.edgecut_weight)

        #     images = [os.path.join(save_path, f"{pref}_hard_poisson_epoch_{self.current_epoch:05}_seam_length_{np.sum(seamlengths):04f}.png")] + \
        #             [os.path.join(save_path, f"distortionloss_{pref}_hard_poisson_epoch_{self.current_epoch:05}_seam_length_{np.sum(seamlengths):04f}.png")]
        #     self.logger.log_image(key=f'{pref} hard poisson', images=images, step=self.current_epoch)

        ### Losses on 3D surfaces
        ### NOTE: mesh is original mesh topology (not soup)

        ogvs = batch_parts["source_V"].detach().cpu().numpy()
        ogfs = batch_parts["ogT"]

        if 'pred_V' in batch_parts.keys():
            # Plot edge cuts
            topo_cutvpairs = source.ogedge_vpairs_nobound[topo_cutedges_stitch]
            cylinderpos = ogvs[topo_cutvpairs]
            cutlen = torch.sum(source.elens_nobound[topo_cutedges_stitch]).item()

            # Stitch cut
            if len(cylinderpos) > 0:
                export_views(ogvs, batch_parts["ogT"], save_path, filename=f"stitchcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                            plotname=f"{pref} Total stitchcut len: {cutlen:0.5f}", cylinders=cylinderpos,
                            outline_width=0.01, cmap = plt.get_cmap('Reds'),
                            device="cpu", n_sample=30, width=200, height=200,
                            vmin=0, vmax=1, shading=False)

            if validation:
                self.log("val cut length", cutlen, logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

        # Weight cut
        images = []
        if "weights" in batch_parts.keys():
            weights = batch_parts['weights']
            topo_cutvpairs = source.ogedge_vpairs_nobound[topo_cutedges_weight]
            cylinderpos = ogvs[topo_cutvpairs]
            cutlen = torch.sum(source.elens_nobound[topo_cutedges_weight]).item()

            if len(cylinderpos) > 0:
                export_views(ogvs, batch_parts["ogT"], save_path, filename=f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                            plotname=f"{pref} Total weight cut len: {cutlen:0.5f}", cylinders=cylinderpos,
                            outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                            device="cpu", n_sample=30, width=200, height=200,
                            vmin=0, vmax=1, shading=False)

            # Weights
            # NOTE: below results in 2x each cylinder but it's fine
            cylinderpos = ogvs[source.ogedge_vpairs_nobound.detach().cpu().numpy()]
            cylindervals = np.stack([weights, weights], axis=1) # E x 2

            export_views(ogvs, batch_parts["ogT"], save_path, filename=f"weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                        plotname=f"{pref} Edge Weights", cylinders=cylinderpos,
                        cylinder_scalars=cylindervals,
                        outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                        device="cpu", n_sample=30, width=200, height=200,
                        vmin=0, vmax=1, shading=False)

            images = [os.path.join(save_path, f"weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png"),
                      os.path.join(save_path, f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png"),
                      ]

            # if os.path.exists(os.path.join(save_path, f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png")):
            #     images.append(os.path.join(save_path, f"weightcuts_mesh_{self.current_epoch:05}_batch{batch_idx}.png"))
            if (self.args.init in ["tutte", "slim"] and self.args.ninit == -1) or \
                (self.current_epoch == 0 and self.args.init):
                boundary_path = os.path.join(save_path, f"boundary_mesh_{self.current_epoch:05}_batch{batch_idx}.png")
                if os.path.exists(boundary_path):
                    images = [boundary_path] + images

            # Initial cuts
            if os.path.exists(os.path.join(source.source_dir, "..", "..", "initcuts.png")):
                images = [os.path.join(source.source_dir, "..", "..", "initcuts.png")] + images

            # GT Cuts
            if os.path.exists(os.path.join(source.source_dir, "..", "..", "gtcuts.png")):
                images = [os.path.join(source.source_dir, "..", "..", "gtcuts.png")] + images

            # Make sure images are not locked before logging them
            from source_njf.utils import wait_for_file, is_locked
            import time

            count = 0
            while any([is_locked(imgpath) for imgpath in images]):
                if count > 10:
                    print("Weight images locked for too long. Skipping logging.")
                    break
                time.sleep(1)
                count += 1

            if count <= 10:
                try:
                    self.logger.log_image(key=f'{pref} pred weight', images=images, step=self.current_epoch)
                except Exception as e:
                    print(e)

        edgecorrespondences = source.edgecorrespondences
        for key, val in lossdict[0].items():
            if "loss" in key: # Hacky way of avoiding aggregated values
                if key == "vertexseploss":
                    valid_pairs = source.valid_pairs
                    separationloss = val

                    # Convert separation loss to per vertex
                    from collections import defaultdict
                    vlosses = defaultdict(np.double)
                    vlosscount = defaultdict(int)
                    for i in range(len(valid_pairs)):
                        pair = valid_pairs[i]
                        vlosses[pair[0]] += separationloss[i]
                        vlosses[pair[1]] += separationloss[i]
                        vlosscount[pair[0]] += 1
                        vlosscount[pair[1]] += 1

                    # NOTE: Not all vertices will be covered in vlosses b/c they are boundary vertices
                    vseplosses = np.zeros(len(batch_parts['pred_V'])) # Soup vs
                    for k, v in sorted(vlosses.items()):
                        vseplosses[k] = v / vlosscount[k]

                    # NOTE: can't let mesh re-export the faces because the indexing will be off
                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}", outline_width=0.01,
                                vcolor_vals=vseplosses,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=0.5, shading=False)

                elif key == "edgecutloss":
                    from collections import defaultdict
                    edgecutloss = val # E x 1
                    cylinderpos = ogvs[source.ogedge_vpairs_nobound[keepidxs].detach().cpu().numpy()]
                    cylindervals = np.stack([edgecutloss, edgecutloss], axis=1) # E x 2

                    ## Also plot edges which are excluded by keepidxs
                    excludeidxs = np.setdiff1d(np.arange(len(source.ogedge_vpairs_nobound)), keepidxs)
                    subcylinders = ogvs[source.ogedge_vpairs_nobound[excludeidxs].detach().cpu().numpy()]

                    if len(subcylinders) == 0:
                        subcylinders = None

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals, outline_width=0.01,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=self.args.edgecut_weight, shading=False,
                                subcylinders = subcylinders)

                elif key == "gtweightloss":
                    from collections import defaultdict
                    cylinderpos = ogvs[source.ogedge_vpairs_nobound.detach().cpu().numpy()]
                    cylindervals = np.stack([val, val], axis=1) # E x 2

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals, outline_width=0.01,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

                elif key == "edgegradloss": # E x 2
                    cylindervals = []
                    cylinderpos = []
                    ogvs = batch_parts["source_V"].detach().cpu().numpy()
                    count = 0
                    for k, v in sorted(edgecorrespondences.items()):
                        # If only one correspondence, then it is a boundary
                        if len(v) == 1:
                            continue
                        cylinderpos.append(ogvs[list(k)])
                        cylindervals.append([np.sum(val[count]), np.sum(val[count])])
                        count += 1

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals, outline_width=0.01,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=0.5, shading=False)

                elif key in ["distortionloss", "gtjloss", "normalloss"]:
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "gtuvloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}",
                                    vcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "invjloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "fliploss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=0.6, shading=False)
                elif key == "intersectionloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                else:
                    continue

        # Log together: 3D surface losses + initial tutte cut
        images = [os.path.join(save_path, f"{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

        # Filter out all renders that dont exist
        images = [imgpath for imgpath in images if os.path.exists(imgpath)]
        self.logger.log_image(key=f'{pref} 3D losses', images=images, step=self.current_epoch)

def custom_collate(data):
    assert len(data) == 1, 'we work on a single batch object'
    return data[0]


def load_network_from_checkpoint(gen, args=None, cpuonly = False):
    if cpuonly:
        map_location={'cuda:0':'cpu'}
        model = MyNet.load_from_checkpoint(checkpoint_path=gen, map_location=map_location)
    else:
        model = MyNet.load_from_checkpoint(checkpoint_path=gen)
    if args is None:
        args = model.args
    # model.eval()
    # below we should add checks for any argument that cannot be changed retroactively for a loaded network
    # note we should also handle things like the encoder having different fields than what we specify
    # (e.g., loading pose params even if we don't want them) but that is too much hassle to check for something
    # that in any case breaks the loaded network

    loaded_normalization = model.get_layer_normalization_type()
    loaded_pointnet_normalization = model.get_pointnet_layer_normalization_type()
    model.args = args
    cur_normalization = model.get_layer_normalization_type()
    cur_pointnet_normalization = model.get_pointnet_layer_normalization_type()

    if cur_normalization != loaded_normalization:
        warnings.warn(
            f"args specify layer normalization of type {cur_normalization}, but network loaded from checkpoint"
            f" has {loaded_normalization} normalization")
    if cur_pointnet_normalization != loaded_pointnet_normalization:
        warnings.warn(
            f"args specify pointnet layer normalization of type {cur_pointnet_normalization}, but network loaded from checkpoint"
            f" has {loaded_pointnet_normalization} normalization")
    return model


def test_main(dedge_ckpt, duv_ckpt, args):
    pl.seed_everything(48, workers=True)

    # Test directory
    save_path = os.path.join(args.outputdir, args.expname)
    Path(save_path).mkdir(exist_ok=True, parents=True)

    if args.overwrite:
        from utils import clear_directory
        clear_directory(save_path)

    ### Load directedge model
    import copy
    dedge_args = copy.deepcopy(args)
    dedge_args.arch = "directedge"
    if args.novelinference:
        dedge_args.gtuvloss = False
        dedge_args.edgeloss = False
    dedge_model = load_network_from_checkpoint(dedge_ckpt, dedge_args)
    dedge_gen = dedge_model.encoder

    ### Load directuv model
    duv_args = copy.deepcopy(args)
    duv_args.arch = "directuv"
    duv_args.init = "tutte"
    if args.novelinference:
        duv_args.gtuvloss = False
        duv_args.edgeloss = False
    # Direct UV loads data from direct edge output predictions
    pred_path = os.path.join(args.outputdir, args.expname, "predictions", "val")
    duv_args.root_dir_test = pred_path
    duv_args.root_dir_train = pred_path
    duv_model = load_network_from_checkpoint(duv_ckpt, duv_args)
    duv_gen = duv_model.encoder

    with open(os.path.join(args.root_dir_test, args.test_file)) as file:
        data = json.load(file)
        pairs_test = data['pairs']

    print("TEST :", len(pairs_test))
    valid_pairs = pairs_test
    logger = WandbLogger(project=args.projectname, name=args.expname, save_dir=os.path.join(args.outputdir, args.expname), log_model=False,
                         offline=args.debug, resume='allow', id = None)

    ################################ TRAINER #############################
    trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=args.precision, log_every_n_steps=200,
                         max_epochs=args.epochs, sync_batchnorm=args.n_devices != 1,
                         check_val_every_n_epoch=args.val_interval,
                         logger=logger,
                         plugins=[SLURMEnvironment(requeue_signal=SIGUSR1)] if not args.debug else None,
                         num_sanity_val_steps=0,
                         enable_model_summary=False,
                         enable_progress_bar=True,
                         num_nodes=1,
                         gradient_clip_val=args.gradclip,
                         deterministic= args.deterministic,)
    ################################ TRAINER #############################
    # Cache directory
    if args.overwritecache:
        from utils import clear_directory
        traincache = os.path.join(args.root_dir_train, "cache")
        testcache = os.path.join(args.root_dir_test, "cache")
        if os.path.exists(traincache):
            clear_directory(traincache)
        if os.path.exists(testcache):
            clear_directory(testcache)

    if isinstance(trainer.precision, int):
        if trainer.precision == 16:
            use_dtype = torch.half
        elif trainer.precision == 32:
            use_dtype = torch.float
        elif trainer.precision == 64:
            use_dtype = torch.double
        else:
            raise Exception("trainer's precision is unexpected value")
    else:
        if "16" in trainer.precision:
            use_dtype = torch.half
        elif "32" in trainer.precision:
            use_dtype = torch.float
        elif "64" in trainer.precision:
            use_dtype = torch.double
        else:
            raise Exception("trainer's precision is unexpected value")

    valid_dataset = DeformationDataset(valid_pairs, dedge_gen.get_keys_to_load(True),
                                    dedge_gen.get_keys_to_load(False), use_dtype, train=False, args=dedge_args)
    valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                shuffle=False, num_workers=0, persistent_workers=0)

    # ================ #
    # model
    # ================ #

    # NOTE: Network not initializing with correct device!!!
    if has_gpu == "gpu":
        dedge_model.to(torch.device("cuda:0"))
        dedge_model.lossfcn.device = torch.device("cuda:0")

        duv_model.to(torch.device("cuda:0"))
        duv_model.lossfcn.device = torch.device("cuda:0")
    else:
        dedge_model.to(torch.device("cpu"))
        dedge_model.lossfcn.device = torch.device("cpu")

        duv_model.to(torch.device("cpu"))
        duv_model.lossfcn.device = torch.device("cpu")

    dedge_model.type(use_dtype)
    dedge_model.lr = args.lr

    duv_model.type(use_dtype)
    duv_model.lr = args.lr

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& TEST &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # Edge prediction
    trainer.test(dedge_model, valid_loader, ckpt_path=dedge_ckpt)

    # Load edge predictions and run UV prediction
    import re
    valid_edge_pairs = []
    # for idx, (source_index, target_index) in enumerate(valid_dataset.source_and_target):
    #     # Naming convention: sourceind_edgecut.obj
    #     research = re.search(r'(.*)_(\d+).obj', source_index)
    #     sourcename = research.group(1) + "_edgecut"
    #     assert os.path.exists(os.path.join(pred_path, f"{source_index}/{sourcename}.obj")), f"Edge prediction not found for {source_index}"
    #     valid_edge_pairs.append([f"{source_index}/{sourcename}.obj", f"{source_index}/{sourcename}.obj"])

    for idx, batch in enumerate(valid_loader):
        # Naming convention: sourceind_edgecut.obj
        source = batch[0]
        sourcename = source.source_ind
        assert os.path.exists(os.path.join(pred_path, f"{sourcename}_{idx}/{sourcename}_edgecut.obj")), f"Edge prediction not found for {sourcename}"
        valid_edge_pairs.append([f"{sourcename}_{idx}/{sourcename}_edgecut.obj", f"{sourcename}_{idx}/{sourcename}_edgecut.obj"])


    valid_dataset = DeformationDataset(valid_edge_pairs, {'samples': True, 'samples_normals': True, 'samples_wks': True},
                                    {}, use_dtype, train=False, args=duv_args)
    valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                shuffle=False, num_workers=0, persistent_workers=0)
    trainer.test(duv_model, valid_loader, ckpt_path=duv_ckpt)

    return