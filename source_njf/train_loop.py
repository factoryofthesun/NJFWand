#defines the network and the train loop
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
import torchvision

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
torch.autograd.set_detect_anomaly(True)

class MyNet(pl.LightningModule):
    '''
    the network
    '''
    def __init__(self, code_dim, args, encoder = None, point_dim=6, verbose=False, face_dim=0, n_edges=None):
        print("********** Some Network info...")
        print(f"********** code dim: {code_dim}")
        print(f"********** centroid dim: {point_dim}")
        super().__init__()
        self.args = args
        self.arch = self.args.arch

        # HACK: normalize init
        if not hasattr(args, "normalizeinit"):
            args.normalizeinit = False

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

        # HACK: gtjacobianloss
        if not hasattr(args, "gtjacobianloss"):
            args.gtjacobianloss = None

        # HACK: overwrite init
        if not hasattr(args, "overwriteinit"):
            args.overwriteinit = False

        # HACK: noplot
        if not hasattr(args, "noplot"):
            args.noplot = False

        # HACK: showtexture
        if not hasattr(args, "showtexture"):
            args.showtexture = False

        # HACK: clipcheckpointing
        if not hasattr(args, "clipcheckpointing"):
            args.clipcheckpointing = False

        # HACK: sdscheckpointing
        if not hasattr(args, "sdscheckpointing"):
            args.sdscheckpointing = False

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
            # Diffusionnet => vertex features => transfmeshcnn (edge features) => MLP => edge weights
            # Diffusionnet => vertex features => face features => MLP => jacobians
            elif self.arch == "fullconv2":
                from diffusionnet import DiffusionNet

                input_dim = point_dim + code_dim
                vertexdim = self.args.vertexdim

                if self.args.fftextrinsic and self.args.extrinsic:
                    from source_njf.utils import FourierFeatureTransform
                    self.fftextrinsic = FourierFeatureTransform(3, self.args.fftextrinsic, self.args.fftscale)
                    input_dim = input_dim + 2 * self.args.fftextrinsic + 6 # [input, sin, cos]

                self.vertexencoder = DiffusionNet(C_in=input_dim, C_out=vertexdim, C_width=128, N_block=4, outputs_at='vertices',
                                                with_gradient_features=True, with_gradient_rotations=True)
                self.vertexencoder2 = DiffusionNet(C_in=input_dim, C_out=vertexdim, C_width=128, N_block=4, outputs_at='vertices',
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
                # Two options: (1) concat v1 with v2 (2) only use v2
                if self.args.fullconv_share:
                    faceinputdim = 2 * vertexdim
                else:
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

                ## Face MLP
                self.facejacobian_mlp = nn.Sequential(nn.Linear(faceinputdim, 128),
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
                if self.args.initweightinput:
                    if self.args.fftweight > 0:
                        from utils import FourierFeatureTransform

                        self.fftweight = FourierFeatureTransform(1, self.args.fftweight, self.args.fftscale)
                        input_dim += 2 * self.args.fftweight + 1

                    else:
                        input_dim += 1

                if self.args.initjinput:
                    if self.args.fftj > 0:
                        from utils import FourierFeatureTransform

                        self.fftj = FourierFeatureTransform(6, self.args.fftj, self.args.fftscale)
                        input_dim += 2 * self.args.fftj + 6

                    else:
                        input_dim += 6

                self.face_decoder = nn.Sequential(nn.Linear(input_dim, 128),
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

            self.__IDENTITY_INIT = self.args.identity
            if self.__IDENTITY_INIT:
                self.face_decoder[-1].bias.data[:].zero_()
                self.face_decoder[-1].weight.data[:9].zero_()

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

            self.__IDENTITY_INIT = self.args.identity
            if self.__IDENTITY_INIT:
                self.face_decoder[-1].bias.data[:].zero_()
                self.face_decoder[-1].weight.data[:9].zero_()

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

            self.__IDENTITY_INIT = self.args.identity
            if self.__IDENTITY_INIT:
                self.face_decoder[-1].bias.data[:].zero_()
                self.face_decoder[-1].weight.data[:9].zero_()
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

        # TODO: INTEGRATE BOTH KAOLIN AND NVDIFFRAST OPTIONS UNDER SINGLE RENDER CLASS
        # TODO: Renderer => (KaolinRenderer, NVDiffrastRenderer)
        #### Renderer stuff ####
        if self.args.rasterizer is not None:
            if self.args.rasterizer == "nvdgl":
                import nvdiffrast.torch as dr
                self.rastercontext = dr.RasterizeGLContext()
            elif self.args.rasterizer == "nvdcuda":
                import nvdiffrast.torch as dr
                self.rastercontext = dr.RasterizeCudaContext()
            elif self.args.rasterizer == "kaolin":
                from source_njf.renderer import Renderer
                self.renderer = Renderer(self.device, dim=(self.args.renderresolution, self.args.renderresolution),
                                         interpolation_mode = 'bilinear')
            else:
                raise ValueError(f"Unknown rasterizer: {self.args.rasterizer}.")

        # if self.args.lowmemory:
        #     device = torch.device('cpu')
        # else:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(self.args.sdsloss) > 0:
            for sds in self.args.sdsloss:
                if sds in ["csd", 'text2img']:
                    from diffusion_guidance.deepfloyd_cascaded import DeepFloydCascaded
                    self.diffusion = DeepFloydCascaded(max_step_percent=self.args.sds_max_step_percent,
                                                    cascaded=True if sds == 'csd' else False,
                                                    device = self.device) # optionally you can pass a config at initialization
                elif sds == "img2img":
                    from diffusion_guidance.deepfloyd_if import DeepFloydIF_Img2Img
                    self.diffusion = DeepFloydIF_Img2Img(device = self.device)
                elif sds == "controlnet":
                    from diffusion_guidance.sd_controlnet import ControlSDS
                    self.diffusion_cnet = ControlSDS(device = self.device, conditioning = self.args.controlcondition)
                else:
                    raise Exception(f"Invalid SDS loss type {sds}.")
        elif self.args.sdscheckpointing:
            from diffusion_guidance.deepfloyd_cascaded import DeepFloydCascaded
            self.diffusion = DeepFloydCascaded(max_step_percent=self.args.sds_max_step_percent,
                                            cascaded=True if self.args.sdsloss == 'csd' else False,
                                            device = self.device) # optionally you can pass a config at initialization

        if self.args.textureimg is not None:
            self.textureimg = {}
            self.gttex = {}
            self.texturenames = []

            for i in range(len(self.args.textureimg)):
                imgpath = self.args.textureimg[i]
                if self.args.texturename is not None:
                    texturename = self.args.texturename[i]
                else:
                    texturename = os.path.basename(imgpath).split('.')[0]

                self.texturenames.append(texturename)
                self.textureimg[texturename] = torchvision.io.read_image(imgpath).float().to(self.device)/255.

                # Set gttex if supervising with ground truth renders
                if self.args.gttex:
                    # texturename => (vertices, faces, uv, uvfaces, reference render resolution)
                    from meshing.io import PolygonSoup

                    objpath = self.args.gttex[i]
                    soup = PolygonSoup.from_obj(objpath)

                    # Center the obj
                    # soup.vertices = soup.vertices - np.mean(soup.vertices, axis=0)
                    # soup.vertices /= np.max(np.linalg.norm(soup.vertices, axis=1))

                    vs = torch.from_numpy(soup.vertices).float().to(self.device)
                    fs = torch.from_numpy(soup.indices.astype(int)).long().to(self.device)
                    uvs = torch.from_numpy(soup.uvs).float().to(self.device)
                    uvfs = torch.from_numpy(soup.face_uv.astype(int)).long().to(self.device)
                    resolution = 2048
                    self.gttex[texturename] = (vs, fs, uvs, uvfs, resolution)

            # Automatically compute max mip levels based on image resolutions
            import math
            self.miplevels = {}
            for texturename in self.textureimg:
                minshape = min(self.textureimg[texturename].shape[1:])
                self.miplevels[texturename] = int(math.log2(minshape))

        if self.args.clipcheckpointing:
            from torchvision import transforms
            import clip

            self.clip_model, self.clip_preprocess = clip.load(
                "RN101", self.device, jit=False)
            self.normalize_transform = transforms.Compose([
                self.clip_preprocess.transforms[0],  # Resize
                self.clip_preprocess.transforms[1],  # CenterCrop
                self.clip_preprocess.transforms[-1],  # Normalize
            ])

            from collections import defaultdict

            self.best_cliploss = defaultdict(lambda: 1e6)
            self.best_cliploss_epoch = defaultdict(int)

        if self.args.sdscheckpointing:
            from collections import defaultdict

            self.best_sdsloss = defaultdict(lambda: 1e6)
            self.best_sdsloss_epoch = defaultdict(int)

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
        elif self.arch == 'fullconv2':
                # Check for fftextrinsics
                if self.args.fftextrinsic and self.args.extrinsic:
                    fftvertices = self.fftextrinsic(source.get_vertices().to(self.device))
                    normals = source.get_loaded_data('vertexnormals')
                    x = torch.cat([x, fftvertices, normals], dim=1)

                vertexcodes = self.vertexencoder(x, source.get_loaded_data('mass'), L=source.get_loaded_data('L'),
                                        evals=source.get_loaded_data('evals'), evecs=source.get_loaded_data('evecs'),
                                        gradX=source.get_loaded_data('gradX'), gradY=source.get_loaded_data('gradY'),
                                        faces=source.get_loaded_data('faces'))
                vertexcodes2 = self.vertexencoder2(x, source.get_loaded_data('mass'), L=source.get_loaded_data('L'),
                                        evals=source.get_loaded_data('evals'), evecs=source.get_loaded_data('evecs'),
                                        gradX=source.get_loaded_data('gradX'), gradY=source.get_loaded_data('gradY'),
                                        faces=source.get_loaded_data('faces'))
                return vertexcodes, vertexcodes2

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

            # Clamp the weights
            if self.args.minweight:
                weights = torch.clamp(weights, max = -self.args.minweight)

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

            if self.args.minweight:
                weights = torch.clamp(weights, max = -self.args.minweight)

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
        elif self.arch == 'fullconv2':
            # Construct edge and face codes from vertex codes
            vertexcodes, vertexcodes2 = res

            ### Edge decoding
            #### Predict edge weights #####
            # NOTE: symmetric edge features (|v1 - v2|, v1 + v2)
            edge_vpairs_nobound = source.ogedge_vpairs_nobound.detach().cpu().numpy()

            # Vertex features => MeshCNN edge features
            meshcnn_edge_vpairs = source.edges
            edgecodes = torch.cat([torch.abs(vertexcodes[meshcnn_edge_vpairs[:,0]] - vertexcodes[meshcnn_edge_vpairs[:,1]]),
                                    vertexcodes[meshcnn_edge_vpairs[:,0]] + vertexcodes[meshcnn_edge_vpairs[:,1]]], dim=1)

            # TODO: cache the meshcnn to njf mapping
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
                weights = -dclamp(edgevals + source.initweights.to(edgevals.device), 1e-9, 1000)

            if self.args.minweight:
                weights = torch.clamp(weights, max = -self.args.minweight)

            faces = source.get_source_triangles()
            edge_vpairs_nobound = source.ogedge_vpairs_nobound

            #### Predict jacobians #####
            if self.args.fullconv_share:
                facecodes = torch.mean(vertexcodes[faces,:], dim=1)
                facecodes2 = torch.mean(vertexcodes2[faces,:], dim=1)
                facecodes = torch.cat([facecodes, facecodes2], dim=1)
            else:
                facecodes = torch.mean(vertexcodes2[faces,:], dim=1)

            # Conditioning
            if self.args.initjinput:
                # Remove the 0 dimension
                initjinput = initj[:, :2, :].reshape(len(facecodes), -1)

                if self.args.fftj > 0:
                    initjinput = self.fftj(initjinput)

                facecodes = torch.cat([facecodes, initjinput], dim=1)

            res = self.facejacobian_mlp(facecodes)

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
            if self.args.softpoisson is not None:
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

                if self.args.minweight:
                    weights = torch.clamp(weights, max = -self.args.minweight)

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

            if self.args.softpoisson is None:
                return ret

            return ret, weights

        ## TODO: REFACTOR BELOW IF GLOBAL CODES ##
        # because of stacking the result is a 9-entry vec for each (z_i|c_j), now let's turn it to a batch x tris x 9 tensor
        if self.arch == "mlp":
            pred_J = net_input.back_to_non_stacked(res)
            ret = pred_J.reshape(pred_J.shape[0], pred_J.shape[1], 3, 3)
        else:
            ret = res.reshape(1, source.mesh_processor.faces.shape[0], 3, 3)

        # if we chose to have the identity as the result when the prediction is 0,
        if self.__IDENTITY_INIT:
            for i in range(0, 3):
                ret[:, :, i, i] += 1

        if self.args.softpoisson or self.args.optweight:
            return ret, weights
        else:
            return ret

    def extract_code(self, source, target):
        '''
		given a batch, extract the global code w.r.t the source and targets, using the set encoders
		:param batch: the batch object
		:return: Bx|z| batch of codes z
		'''
        return self.encoder.encode_deformation(source, target)

    ### Run hardpoisson based on the network predictions
    ### - edge cuts: based on the soft poisson UVs
    ### - Compute cut lengths, distortion
    # Type: uv = UV distance for cuts; weight = weights for cuts
    def hardpoisson(self, source, pred_uvs, pred_jacobians, weights, cuttype = 'uv', threshold=0.01, edgeseparation=None):
        # NOTE: Predj = (F x 2 x 3)
        elens = source.elens_nobound.detach().cpu().numpy()
        weights = weights
        vertices = source.get_vertices().detach().cpu().numpy()
        faces = source.get_source_triangles()

        from igl import cut_mesh
        ogedge_vpairs_nobound = source.ogedge_vpairs_nobound.detach().cpu().numpy()

        if cuttype == 'uv':
            if edgeseparation is None:
                edge_vpairs = source.edge_vpairs.detach().cpu().numpy()
                uvpairs = pred_uvs[edge_vpairs, :] # E x 2 x 2 x 2
                edgeseparation = torch.sum(torch.nn.functional.l1_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2)
                edgeseparation = torch.mean(edgeseparation, dim=1).detach().cpu().numpy() # E x 1
            assert len(ogedge_vpairs_nobound) == len(edgeseparation), f"ogedge_vpairs_nobound len: {len(ogedge_vpairs_nobound)}, edge separation len: {len(edgeseparation)}"
            cutedges = np.where(edgeseparation > threshold)[0]
            cutpairs = [list(ogedge_vpairs_nobound[e]) for e in cutedges]
        elif cuttype == "weight":
            assert len(ogedge_vpairs_nobound) == len(weights), f"ogedge_vpairs_nobound len: {len(ogedge_vpairs_nobound)}, weights len: {len(weights)}"

            cutedges = np.where(weights > threshold)[0]
            cutpairs = [list(ogedge_vpairs_nobound[e]) for e in cutedges]
        elif cuttype == "jacobians":
            fpairs_nobound = source.facepairs_nobound
            jacob_diff = torch.linalg.matrix_norm(pred_jacobians[fpairs_nobound[:,0]] - pred_jacobians[fpairs_nobound[:,1]], ord='fro', dim=(1,2)).detach().cpu().numpy()
            cutedges = np.where(jacob_diff > threshold)[0]
            cutpairs = [list(ogedge_vpairs_nobound[e]) for e in cutedges]

        # Cut length
        cut_length = np.sum(elens[cutedges])

        if len(cutedges) > 0:
            cutfes = np.zeros((len(faces), 3)).astype(int)
            edges_list = []
            for f in faces:
                for i in range(3):
                    e = [f[i], f[(i+1)%3]]
                    edges_list.append(e)

            for i in range(len(edges_list)):
                edgecheck = edges_list[i]
                if edgecheck in cutpairs or edgecheck[::-1] in cutpairs:
                    cutfes[i // 3, i % 3] = 1
            cutvs, cutfs = cut_mesh(vertices, faces, cutfes)
        else:
            cutvs = vertices
            cutfs = faces

        ### NOTE: If multiple connected components: then we compute poisson solve + distortion for each one!!
        from igl import connected_components, adjacency_matrix, flipped_triangles
        A = adjacency_matrix(cutfs).astype(int)
        num, comps, k = connected_components(A)

        if num > 1:
            final_uvs = []
            distortion = []
            cutvs_list = []
            cutfs_list = []
            fidx_list = []

            newmesh = Mesh(cutvs, cutfs)
            fgroups = np.zeros(len(newmesh.faces)).astype(int)
            for i in np.unique(comps):
                compi_vertices = np.where(comps == i)[0]

                # Get all faces associated with these vertices
                compi_faces = []
                for fi in range(len(newmesh.faces)):
                    f = newmesh.faces[fi]
                    if np.isin(f, compi_vertices).any():
                        assert np.isin(f, compi_vertices).all()
                        compi_faces.append(fi)

                fgroups[compi_faces] = i

            # Sort fgroups by size in case we need to truncate
            unique_fgroups, counts = np.unique(fgroups, return_counts=True)
            unique_fgroups = unique_fgroups[np.argsort(counts)[::-1]]
            for i in unique_fgroups:
                compi_faces = np.where(fgroups == i)[0]
                fidx_list.append(compi_faces)
                subvs, subfs = newmesh.export_submesh(compi_faces)

                cutvs_list.append(subvs)
                cutfs_list.append(subfs)

                subvs = torch.from_numpy(subvs).double().to(self.device)
                subfs = torch.from_numpy(subfs).long().to(self.device)

                ## Run poisson solve on cut mesh
                grad = igl.grad(subvs.detach().cpu().numpy(), subfs.detach().cpu().numpy())
                d_area = igl.doublearea(subvs.detach().cpu().numpy(), subfs.detach().cpu().numpy())
                d_area = np.hstack((d_area, d_area, d_area)) # This matches the format for grad matrix (t0, t1, t2, ..., t0, t1, t2, ..., t0, t1, t2, ...)
                mass = np.diag(d_area)
                rhs = grad.T@mass
                laplace = grad.T@mass@grad

                ## Update diagonal of Laplacian such that rows sum to 0
                np.fill_diagonal(laplace, 0)
                laplace[range(len(laplace)), range(len(laplace))] = -np.sum(laplace, axis=1)
                np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)
                grad = grad.todense()

                ## Convert to torch tensors
                laplace = torch.from_numpy(laplace).double().to(self.device)
                rhs = torch.from_numpy(rhs).double().to(self.device)

                # Triangle soup solve
                input = pred_jacobians[compi_faces].transpose(2, 1).reshape(1, -1, 2) # 1 x F*3 x 2

                # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
                P = torch.zeros(input.shape, dtype=input.dtype).to(self.device)
                k = input.shape[1] // 3
                P[:, :k, :] = input[:, ::3] # First row of all jacobians together
                P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
                P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together
                input_to_solve = rhs @ P

                # Poisson solve
                sub_uvs = torch.linalg.solve(laplace[1:,1:], input_to_solve[0, 1:]) # V x 2
                sub_uvs = torch.cat([torch.zeros(1, 2, device=self.device), sub_uvs], dim=0) # V x 2
                final_uvs.append(sub_uvs.detach().cpu().numpy())

                ## Get jacobians
                from source_njf.utils import get_jacobian_torch
                hardpoisson_jacobians = get_jacobian_torch(subvs, subfs, sub_uvs, device=self.device)

                ## Compute distortion energy
                from source_njf.losses import symmetricdirichlet
                subdistortion = symmetricdirichlet(subvs, subfs, hardpoisson_jacobians).detach().cpu().numpy()
                distortion.append(subdistortion)

            return final_uvs, cutedges, cutpairs, distortion, cut_length, cutvs_list, cutfs_list, fidx_list
        else:
            cutvs = torch.from_numpy(cutvs).double().to(self.device)
            cutfs = torch.from_numpy(cutfs).long().to(self.device)

            ## Run poisson solve on cut mesh
            grad = igl.grad(cutvs.detach().cpu().numpy(), cutfs.detach().cpu().numpy())
            d_area = igl.doublearea(cutvs.detach().cpu().numpy(), cutfs.detach().cpu().numpy())
            d_area = np.hstack((d_area, d_area, d_area)) # This matches the format for grad matrix (t0, t1, t2, ..., t0, t1, t2, ..., t0, t1, t2, ...)
            mass = np.diag(d_area)
            rhs = grad.T@mass
            laplace = grad.T@mass@grad

            ## Update diagonal of Laplacian such that rows sum to 0
            np.fill_diagonal(laplace, 0)
            laplace[range(len(laplace)), range(len(laplace))] = -np.sum(laplace, axis=1)
            np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)
            grad = grad.todense()

            ## Convert to torch tensors
            laplace = torch.from_numpy(laplace).double().to(self.device)
            rhs = torch.from_numpy(rhs).double().to(self.device)

            # Triangle soup solve
            input = pred_jacobians.transpose(2, 1).reshape(1, -1, 2) # 1 x F*3 x 2

            # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
            P = torch.zeros(input.shape, dtype=input.dtype).to(self.device)
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3] # First row of all jacobians together
            P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
            P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together
            input_to_solve = rhs @ P

            # Poisson solve
            final_uvs = torch.linalg.solve(laplace[1:,1:], input_to_solve[0, 1:]) # V x 2
            final_uvs = torch.cat([torch.zeros(1, 2, device=self.device), final_uvs], dim=0) # V x 2
            assert len(final_uvs) == len(cutvs), f"final_uvs len: {len(final_uvs)}, cutvs len: {len(cutvs)}"

            ## Get jacobians
            from source_njf.utils import get_jacobian_torch
            hardpoisson_jacobians = get_jacobian_torch(cutvs, cutfs, final_uvs, device=self.device)

            ## Compute distortion energy
            from source_njf.losses import symmetricdirichlet
            distortion = symmetricdirichlet(cutvs, cutfs, hardpoisson_jacobians).detach().cpu().numpy()

            return final_uvs.detach().cpu().numpy(), cutedges, cutpairs, distortion, cut_length, cutvs.detach().cpu().numpy(), cutfs.detach().cpu().numpy(), []

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

        if self.current_epoch % self.args.val_interval == 0 and not self.args.noplot:
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
        # NOTE: HACK
        if not hasattr(self.args, "gtedgeloss"):
            self.args.gtedgeloss = None

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

    def predict_map(self, source, target, initj=None, inituv=None):
        if self.args.softpoisson or self.args.optweight:
            pred_J, weights = self.predict_jacobians(source, target, initj)

            if torch.any(torch.isnan(pred_J)):
                raise Exception("NaNs in predicted Jacobians.")

            if torch.any(torch.isnan(weights)):
                raise Exception("NaNs in predicted weights.")

            if torch.any(torch.isinf(weights)):
                raise Exception("Infs in predicted weights.")

            if torch.any(torch.isinf(pred_J)):
                raise Exception("Infs in predicted Jacobians.")

        else:
            pred_J = self.predict_jacobians(source, target, initj)

            if torch.any(torch.isnan(pred_J)):
                raise Exception("NaNs in predicted Jacobians.")

            if torch.any(torch.isinf(pred_J)):
                raise Exception("Infs in predicted Jacobians.")

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

        if self.args.gtjacobianloss:
            pred_V = torch.zeros(pred_J.shape, device=self.device)
            pred_J_poiss = torch.zeros(pred_J.shape, device=self.device)
            pred_J_restricted_poiss = torch.zeros(pred_J.shape, device=self.device)
        else:
            pred_V = source.vertices_from_jacobians(pred_J, updatedlap = updatedlap)

            # Get back jacobians from predicted vertices
            # TODO: For now we use pred_J for the predicted jacobians, but we could also use the restriction.
            # My intuition is that using pred_J while spoofing restriction (by dropping the 3rd row) is better for NN since basis
            # restriction can result in pretty arbitrary transformation across neighboring triangles but TBD.

            pred_J_poiss = source.poisson.jacobians_from_vertices(pred_V)
            pred_J_restricted_poiss = source.restrict_jacobians(pred_J)

        # Undo the pinning translation if set
        # NOTE: This assumes the pinned vertex of every component is the first vertex
        if self.args.undopin and inituv is not None:
            pinidxs = source.poisson.lap_pinned
            for i in range(len(pinidxs)):
                pidx = pinidxs[i]
                componenti = source.poisson.components[pidx]
                cidxs = np.where(source.poisson.components == componenti)[0]
                pred_V[0, cidxs, :2] += inituv[pidx]

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

        val_loss = batch_parts['loss'].item()
        self.log('val_loss', val_loss, logger=True, prog_bar=True, batch_size=1, on_epoch=True, on_step=False)
        self.val_step_iter += 1

        if torch.rand(1).item() > self.args.valrenderratio or self.args.noplot:
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
        import torch
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
            inituv = source.isofuv.squeeze().to(self.device)
        elif self.args.init == "slim":
            sourcedim = 2
            initj = source.slimj.squeeze().to(self.device)
            inituv = source.slimuv.squeeze().to(self.device)
        elif self.args.init == "precut":
            sourcedim = 2
            initj = source.prej.squeeze().to(self.device)
            inituv = source.preuv.squeeze().to(self.device)

        # Need to export mesh soup to get correct face to tutte uv indexing
        vertices = source.get_vertices().to(self.device)
        faces = torch.from_numpy(source.get_source_triangles()).long().to(self.device)

        pred_V = pred_J = pred_J_poiss = pred_J_restricted_poiss = weights = None
        if self.arch == "directuv":
            preduv = self.predict_uv(source, target, inituv = inituv[:,:2])
        elif self.arch == "directedge":
            predcuts = self.predict_edge(source, target)
        else:
            if self.args.softpoisson or self.args.optweight:
                pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss, weights = self.predict_map(source, target, initj=initj if initj is not None else None,
                                                                                                  inituv=inituv)
            else:
                pred_V, pred_J, pred_J_poiss, pred_J_restricted_poiss = self.predict_map(source, target, initj=initj if initj is not None else None,
                                                                                         inituv=inituv)

            # Drop last dimension of restricted J
            if pred_J_restricted_poiss.shape[2] == 3:
                pred_J_restricted_poiss = pred_J_restricted_poiss[:,:,:2]

        if source.fixedvs is not None:
            pred_V[0, source.fixedvs, :2] = inituv[source.fixedvs]

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

        # NOTE: If no softpoisson, then need to convert pred V to soup topology before loss and visualization
        if self.args.softpoisson is None:
            pred_V = pred_V[:, source.faces, :2].reshape(-1,  3, 2)
        else:
            pred_V = pred_V[:, :, :2].squeeze().reshape(-1, 3, 2)

        # NOTE predict_map already composites pred_J against initj
        if self.arch == "directuv":
            loss = self.lossfcn.computeloss(vertices, faces, preduv.reshape(-1, 3, 2),
                                            weights=weights, stitchweights=self.stitchweights[batch_idx],
                                            source=source, keepidxs=source.keepidxs,
                                            normalgrid=normalgrid)
        elif self.arch == "directedge":
            loss = self.lossfcn.computeloss(vertices, faces,weights=predcuts, source=source,)
        else:
            loss = self.lossfcn.computeloss(vertices, faces, ZeroNanGrad.apply(pred_V), ZeroNanGrad.apply(pred_J_poiss[:,:,:2,:]),
                                            weights=weights, stitchweights=self.stitchweights[batch_idx],
                                            source=source, keepidxs=source.keepidxs, predj = pred_J,
                                            normalgrid=normalgrid)

        lossrecord = self.lossfcn.exportloss()
        self.lossfcn.clear() # This resets the loss record dictionary

        ### ==== Visual Losses ==== ###
        import numpy as np

        if len(self.args.visualloss) > 0:
            import torch

            # Set cache directories
            # import os
            # os.environ['HF_DATASETS_CACHE'] = "/net/scratch/rliu/.cache"
            # os.environ['HF_HOME'] = "/net/scratch/rliu/.cache"

            # Low memory hacks
            if self.args.lowmemory:
                if "controlnet" in self.args.sdsloss:
                    self.diffusion_cnet.pipe = self.diffusion_cnet.pipe.to(torch.device("cpu"))
                    self.diffusion_cnet.alphas = self.diffusion_cnet.alphas.to(torch.device("cpu"))
                    self.diffusion_cnet.device = torch.device("cpu")

                if "text2img" in self.args.sdsloss or "csd" in self.args.sdsloss:
                    self.diffusion.stage_I_alphas = self.diffusion.stage_I_alphas.to(torch.device("cpu"))
                    self.diffusion.stage_I_pipe = self.diffusion.stage_I_pipe.to(torch.device("cpu"))

                    if "csd" in self.args.sdsloss:
                        self.diffusion.stage_II_alphas = self.diffusion.stage_II_alphas.to(torch.device("cpu"))
                        self.diffusion.stage_II_pipe = self.diffusion.stage_II_pipe.to(torch.device("cpu"))

                    self.diffusion.device = torch.device("cpu")

            # HACK: Stupid pytorch lightning wants us to make these float weights
            # if "controlnet" in self.args.sdsloss:
            #     self.diffusion_cnet.pipe.to(dtype=torch.float16)

            # if "text2img" in self.args.sdsloss or "csd" in self.args.sdsloss:
            #     self.diffusion.stage_I_pipe.to(dtype=torch.float16)

            #     if "csd" in self.args.sdsloss:
            #         self.diffusion.stage_II_pipe.to(dtype=torch.float16)

            ### Render ###
            # Sample elev/azim
            # Elev: 0 to pi/2, Azim: 0 to 2pi
            elev = ((torch.rand(self.args.nviews) - 0.5) * 2 * 0.5 * np.pi/4).float().to(self.device)
            azim = (torch.rand(self.args.nviews) * 2 * np.pi).float().to(self.device)

            for texturename in source.texturenames:
                if texturename not in self.textureimg:
                    continue

                if texturename not in source.texturedescrip:
                    continue

                lossrecord[0][f'{texturename}_predrenders'] = []

                tex = self.textureimg[texturename].float().to(self.device)
                mip = self.miplevels[texturename]

                # TODO: Need vertexnormals for source mesh
                if self.args.rasterizer == "kaolin":
                    self.renderer.device = self.device
                    self.renderer.camera_projection = self.renderer.camera_projection.to(self.device)
                    self.renderer.lights = self.renderer.lights.to(self.device)
                    self.renderer.background = self.renderer.background.to(self.device)

                    if self.args.up == 'x':
                        up = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
                    elif self.args.up == 'y':
                        up = torch.tensor([0.0, 1.0, 0.0]).to(self.device)
                    else:
                        up = torch.tensor([0.0, 0.0, 1.0]).to(self.device)

                    # Debugging: check behavior when out of bounds
                    if self.args.debug:
                        pred_V += 1

                    ## Generate random additional lights if set
                    add_l_azim = []
                    add_l_elev = []
                    if self.args.randomlights:
                        n_add_lights = np.random.randint(self.args.minaddlights, self.args.maxaddlights + 1)
                        add_l_elev = list(np.random.uniform(-np.pi/2, np.pi/2, n_add_lights))
                        add_l_azim = list(np.random.uniform(0, 2 * np.pi, n_add_lights))

                    l_elev = [0] * 4 + [np.pi/2, -np.pi/2] + add_l_elev
                    l_azim = list(np.linspace(0, 2 * np.pi, 4)) + [0, 0] + add_l_azim

                    pred_render, pred_mask = self.renderer.render_texture(vertices.float(), faces.long(), pred_V.float(),
                                                                            tex, elev=elev, azim=azim, white_background=True,
                                                                            specular=self.args.specular,
                                                                            l_elev = l_elev, l_azim = l_azim,
                                                                            up = up, radius=2.7, mod=True,)

                for visualloss in self.args.visualloss:
                    if visualloss == "sds":
                        from source_njf.renderer import render, random_rotation_translation, translate, projection

                        ##### SDS #####
                        for sdsloss in self.args.sdsloss:
                            if sdsloss == "controlnet":

                                if self.args.controlcondition == "canny":
                                    # NOTE: Canny edge is bad with the textured render! Use just flat render instead (or with good lighting)
                                    colors = torch.ones((len(vertices), 3)).to(self.device).float() * 0.7
                                    control_render, mask = self.renderer.render_mesh(vertices.float(), faces.long(), colors,
                                                                        elev=elev, azim=azim, white_background=True,
                                                                        up = up, radius=2.7)

                                    import cv2
                                    import numpy as np

                                    low_threshold = 100
                                    high_threshold = 200
                                    image = (control_render.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
                                    control_image = []
                                    for i in range(image.shape[0]):
                                        tmp = cv2.Canny(image[i], low_threshold, high_threshold)
                                        control_image.append(tmp)

                                    # Stack and back to torch image and add channel dimension
                                    control_image = torch.from_numpy(np.stack(control_image)).to(self.device).unsqueeze(-1).repeat(1, 1, 1, 3)
                                    control_image = control_image.permute(0, 3, 1, 2).float() / 255.0

                                elif self.args.controlcondition == "depth":
                                    control_render, mask = self.renderer.render_depth(vertices.float(), faces.long(),
                                                                        elev=elev, azim=azim, up = up, radius=2.7)
                                    control_image = control_render

                                if self.args.debug:
                                    import torchvision

                                    tmp = control_image.detach().cpu()
                                    for i, img in enumerate(tmp):
                                        img = torchvision.transforms.functional.to_pil_image(img)
                                        img.save(f"./outputs/scratch/denoise{i}_controlimg.png")

                                if self.args.lowmemory:
                                    self.diffusion_cnet.pipe = self.diffusion_cnet.pipe.to(self.device)
                                    self.diffusion_cnet.alphas = self.diffusion_cnet.alphas.to(self.device)
                                    self.diffusion_cnet.device = self.device

                                texturetext = source.texturedescrip[texturename]
                                text_z, text_z_neg = self.diffusion_cnet.encode_prompt(texturetext)

                                sds = self.diffusion_cnet(pred_render[:,:3], control_image=control_image, prompt_embeds=text_z,
                                                    negative_embeds = text_z_neg)
                                loss_sds = self.args.control_weight * sds['loss_sds']

                                loss += loss_sds
                                lossrecord[0][f'{texturename}_sdcontrolloss'] = self.args.control_weight * sdsloss.cpu().detach().numpy()
                                lossrecord[0][f'{texturename}_target'] = sds['target'].cpu().detach().numpy()

                                # Debugging
                                if self.args.debug:
                                    import torchvision

                                    # Generate inference imgs
                                    import torch.nn.functional as F
                                    with torch.no_grad():
                                        rgb_BCHW_512 = F.interpolate(pred_render[:,:3].detach(), (512, 512), mode="bilinear", align_corners=False)
                                        latents = self.diffusion_cnet.encode_latent_images(rgb_BCHW_512)
                                        noise = torch.randn_like(latents)
                                        latents = self.diffusion_cnet.scheduler.add_noise(latents, noise, torch.tensor(self.diffusion_cnet.max_step, device=self.device))

                                        denoise_image = self.diffusion_cnet.full_denoise(latents, [texturetext] * len(latents), control_image=control_image)

                                        # Save list of denoised imgs
                                        for i, img in enumerate(denoise_image):
                                            img.save(f"./outputs/scratch/denoise{i}.png")

                            else:
                                if self.args.lowmemory:
                                    self.diffusion.stage_I_alphas = self.diffusion.stage_I_alphas.to(self.device)
                                    self.diffusion.stage_I_pipe = self.diffusion.stage_I_pipe.to(self.device)

                                    if sdsloss == 'csd':
                                        self.diffusion.stage_II_alphas = self.diffusion.stage_II_alphas.to(self.device)
                                        self.diffusion.stage_II_pipe = self.diffusion.stage_II_pipe.to(self.device)

                                    self.diffusion.device = self.device

                                texturetext = source.texturedescrip[texturename]
                                text_z, text_z_neg = self.diffusion.encode_prompt(texturetext)

                                sds = self.diffusion(pred_render[:,:3], mask_img = pred_mask, prompt_embeds=text_z, stage='I')
                                loss_sds = sds['loss_sds']

                                loss += loss_sds
                                lossrecord[0][f'{texturename}_sdsloss_stage1'] = loss_sds.cpu().detach().numpy()
                                lossrecord[0][f'{texturename}_target_stage1'] = sds['target'].cpu().detach().numpy()

                                # Stage 2 if cascaded
                                if sdsloss == 'csd':
                                    sds = self.diffusion(pred_render[:,:3], mask_img = pred_mask, prompt_embeds=text_z, stage='II')

                                    # Stage 2 on a lerp schedule
                                    s2_weight = self.args.csd_minweight + (self.args.csd_maxweight - self.args.csd_minweight) * (self.current_epoch/self.trainer.max_epochs)
                                    loss_sds = s2_weight * sds['loss_sds']

                                    loss += loss_sds
                                    lossrecord[0][f'{texturename}_sdsloss_stage2'] = s2_weight * loss_sds.cpu().detach().numpy()
                                    lossrecord[0][f'{texturename}_target_stage2'] = sds['target'].cpu().detach().numpy()

                                if self.args.debug:
                                    import torchvision

                                    # Generate inference imgs
                                    import torch.nn.functional as F
                                    with torch.no_grad():
                                        denoise_image = self.diffusion.full_denoise(pred_render[:,:3], prompt_embeds = text_z,
                                                                                    negative_embeds = text_z_neg)

                                        if sdsloss == "csd":
                                            denoise_image = self.diffusion.full_denoise(denoise_image[:,:3],
                                                                                        prompt_embeds = torch.cat([text_z] * self.args.nviews),
                                                                                        negative_embeds = torch.cat([text_z_neg] * self.args.nviews), stage="II")

                                        # Post processing
                                        denoise_image = (denoise_image / 2 + 0.5).clamp(0, 1)
                                        denoise_image = denoise_image.cpu().detach().float()

                                        # Save list of denoised imgs
                                        for i, img in enumerate(denoise_image):
                                            img = torchvision.transforms.functional.to_pil_image(img)
                                            img.save(f"./outputs/scratch/denoise{i}.png")

                        # Save the render images for visualization
                        import torchvision

                        for i in range(self.args.nviews):
                            predrender = torchvision.transforms.functional.to_pil_image(pred_render[i].cpu().detach())
                            lossrecord[0][f'{texturename}_predrenders'].append(predrender)

                    if visualloss == "render":
                        # TODO: supervise with ground truth render
                        from source_njf.renderer import render, random_rotation_translation, translate, projection

                        if 'renderloss' not in lossrecord[0].keys():
                            lossrecord[0]['renderloss'] = []

                        # Kaolin options
                        if self.args.rasterizer == "kaolin":
                            # NVD Options
                            # TODO: Per-mesh lighting stuff, views, background, etc ...

                            # Sample elev/azim
                            # Elev: 0 to pi/2, Azim: 0 to 2pi
                            elev = ((torch.rand(self.args.nviews) - 0.5) * 2 * 0.5 * np.pi/4).float().to(self.device)
                            azim = (torch.rand(self.args.nviews) * 2 * np.pi).float().to(self.device)
                        else:
                            # Sample random view
                            RADIUS = 3.5
                            r_rot = random_rotation_translation(0.25)
                            r_mv = np.matmul(translate(0, 0, -RADIUS), r_rot)

                            # Projection matrix
                            proj_mtx = projection(x=0.4, f=1000.0)

                            # MVP matrix is just product
                            mvp = torch.from_numpy(np.matmul(proj_mtx, r_mv)).float().to(self.device)

                        # TODO: Loop through the textures assigned for this mesh, load up the relevant references
                        # TODO: Compute masked L2 loss over the relevant pixels
                        for texturename in source.texturenames:
                            if texturename not in self.textureimg:
                                continue

                            lossrecord[0][f'{texturename}_gtrenders'] = []
                            lossrecord[0][f'{texturename}_predrenders'] = []

                            vs, fs, uvs, uvfs, resolution = self.gttex[texturename]
                            tex = self.textureimg[texturename].to(self.device)
                            mip = self.miplevels[texturename]

                            # NOTE: We need to center in the same way as the source mesh
                            vs = vs.float().to(self.device) - torch.from_numpy(source.get_mesh_centroid()).float().to(self.device)
                            fs = fs.int().to(self.device)
                            uvs = uvs.float().to(self.device)
                            uvfs = uvfs.int().to(self.device)
                            tex = tex.float().to(self.device)

                            # TODO: Need vertexnormals for gt and source mesh

                            if self.args.rasterizer == "kaolin":
                                uv_face = uvs[uvfs]
                                self.renderer.device = self.device
                                self.renderer.camera_projection = self.renderer.camera_projection.to(self.device)
                                self.renderer.lights = self.renderer.lights.to(self.device)
                                self.renderer.background = self.renderer.background.to(self.device)

                                # HACK: Set up direction based on each mesh orientation
                                # TODO: Need normal map sampling

                                if self.args.up == 'x':
                                    up = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
                                elif self.args.up == 'y':
                                    up = torch.tensor([0.0, 1.0, 0.0]).to(self.device)
                                else:
                                    up = torch.tensor([0.0, 0.0, 1.0]).to(self.device)

                                reference_render, reference_mask = self.renderer.render_texture(vs, fs.long(), uv_face, tex, elev=elev,
                                                                                                azim=azim, white_background=True,
                                                                                                up = up, radius=2.7)
                                pred_render, pred_mask = self.renderer.render_texture(vertices.float(), faces.long(), pred_V.float(),
                                                                                    tex, elev=elev, azim=azim, white_background=True,
                                                                                    up = up, radius=2.7)
                            else:
                                # Render references
                                reference_render = render(self.rastercontext, mvp, vs, fs, uvs, uvfs, tex, resolution, True, mip)

                                # Render predictions
                                # Face UVs
                                uv_face = pred_V # F x 3 x 2

                                # Need to scale UVs between 0-1
                                # uv_face = uv_face - torch.min(uv_face.reshape(-1, 2), dim=0)[0]
                                # uv_face = uv_face/torch.max(uv_face)
                                uv_face = uv_face.reshape(-1, 2)
                                fuv = torch.arange(len(uv_face)).reshape(-1, 3).to(self.device)

                                pred_render = render(self.rastercontext, mvp, vertices, faces, uv_face, fuv, tex, 512, True, mip)

                            # if self.args.debug:
                            #     # Save the render images for debugging
                            #     from PIL import Image
                            #     import torchvision

                            #     for i in range(self.args.nviews):
                            #         refrender = torchvision.transforms.functional.to_pil_image(reference_render[i].cpu().detach())
                            #         predrender = torchvision.transforms.functional.to_pil_image(pred_render[i].cpu().detach())

                            #         refrender.save(f"./outputs/scratch/refrender_{texturename}_{self.current_epoch}_{i}.png")
                            #         predrender.save(f"./outputs/scratch/predrender_{texturename}_{self.current_epoch}_{i}.png")

                            # Resize for supervision
                            import torchvision
                            if self.args.resolution != self.args.renderresolution:
                                pred_render_sup = torchvision.transforms.functional.resize(pred_render, (self.args.resolution, self.args.resolution), antialias=True)
                                reference_render_sup = torchvision.transforms.functional.resize(reference_render, (self.args.resolution, self.args.resolution), antialias=True)
                            else:
                                pred_render_sup = pred_render
                                reference_render_sup = reference_render

                            # NOTE: Renderloss B x V x W x H
                            renderloss = torch.sqrt(torch.sum(torch.nn.functional.mse_loss(pred_render_sup, reference_render_sup, reduction="none"),
                                                            dim=1))
                            loss += torch.mean(renderloss)
                            lossrecord[0]['renderloss'].append(renderloss.cpu().detach().numpy())

                            # Save the render images for visualization
                            import torchvision

                            for i in range(self.args.nviews):
                                refrender = torchvision.transforms.functional.to_pil_image(reference_render[i].cpu().detach())
                                predrender = torchvision.transforms.functional.to_pil_image(pred_render[i].cpu().detach())

                                lossrecord[0][f'{texturename}_gtrenders'].append(refrender)
                                lossrecord[0][f'{texturename}_predrenders'].append(predrender)

                    # TODO: UPDATE BELOW WITH NEW RENDER FCN
                    # if visualloss == "image":
                    #     from PIL import Image
                    #     from torchvision.transforms.functional import pil_to_tensor
                    #     from torchvision.transforms import Resize
                    #     import torchvision
                    #     from source_njf.utils import normalize_uv

                    #     vizuv = pred_V.clone()
                    #     with torch.no_grad():
                    #         normalize_uv(vizuv)
                    #         # Scale back to centered at 0.5
                    #         vizuv += 0.5

                    #     texturename = os.path.basename(self.args.textureimg).split(".")[0]
                    #     img = Image.open(self.args.textureimg)
                    #     img = img.convert("RGB")
                    #     textureimg = pil_to_tensor(img).double().to(self.device)

                    #     rgb_images = []

                    #     from source_njf.renderer import render_texture
                    #     total_views = 5
                    #     radius = 2.5
                    #     center = torch.zeros(2)
                    #     azim = torch.linspace(center[0], 2 * np.pi + center[0], total_views + 1)[
                    #         :-1].double().to(self.device)
                    #     elev = torch.zeros(len(azim), device=self.device).double()

                    #     pred_images = render_texture(vertices.double(), faces, vizuv, elev, azim, radius, textureimg/255, lights=None,
                    #                                             resolution=(self.args.renderresolution, self.args.renderresolution), device=self.device, lookatheight=0, whitebg=True,
                    #                                             interpolation_mode = self.args.interpmode)
                    #     pred_images = Resize((self.args.resolution, self.args.resolution))(pred_images)
                    #     rgb_images.append(pred_images)

                    #     gt_images = []
                    #     for i in range(total_views):
                    #         gt_image = torchvision.io.read_image(self.args.imageloss + f"_{i}.png").double().to(self.device)
                    #         gt_image = Resize((self.args.resolution, self.args.resolution))(gt_image)/255
                    #         gt_images.append(gt_image)

                    #     gt_images = torch.stack(gt_images, dim=0)
                    #     imageloss = torch.nn.functional.mse_loss(rgb_images[0]['image'], gt_images, reduction="none")
                    #     loss += torch.mean(imageloss)
                    #     lossrecord[0]['imageloss'].append(imageloss.cpu().detach().numpy())

            # HACK: Stupid pytorch lightning wants us to make these float weights
            # if "controlnet" in self.args.sdsloss:
            #     self.diffusion_cnet.pipe.to(dtype=torch.float32)

            # if "text2img" in self.args.sdsloss or "csd" in self.args.sdsloss:
            #     self.diffusion.stage_I_pipe.to(dtype=torch.float32)

            #     if "csd" in self.args.sdsloss:
            #         self.diffusion.stage_II_pipe.to(dtype=torch.float32)

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
                "loss": loss,
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
                "loss": loss,
            }

            return ret

        elif self.args.gtjacobianloss:
            ret = {
                "target_V": vertices.detach(),
                "source_V": vertices.detach(),
                "pred_J": pred_J.detach(),
                "T": faces.detach().cpu().numpy(),
                "ogT": faces.detach().cpu().numpy(),
                'source_ind': source.source_ind,
                'target_inds': target.target_inds,
                "lossdict": lossrecord,
                "loss": loss,
                "pred_J_poiss": pred_J_poiss[:,:,:2,:].squeeze(),
            }
        else:
            ret = {
                "target_V": vertices.detach(),
                "source_V": vertices.detach(),
                "pred_V": pred_V.detach(),
                "pred_J": pred_J.detach(),
                "T": faces.detach().cpu().numpy(),
                'source_ind': source.source_ind,
                'target_inds': target.target_inds,
                "lossdict": lossrecord,
                "loss": loss,
                "pred_J_poiss": pred_J_poiss[:,:,:2,:].squeeze(),
            }

            if self.args.softpoisson or self.args.optweight or self.args.gtjacobianloss:
                ret['weights'] = weights.detach().cpu().numpy()

            # Need to adjust the return values if no poisson solve
            if len(pred_V) == len(faces):
                ret['pred_V'] = pred_V.detach().reshape(-1, 2)

                # Triangle soup
                ret['ogT'] = ret['T'] # Save original triangle indices
                ret['T'] = np.arange(len(faces)*3).reshape(len(faces), 3)

                # Debugging: predicted fuvs make sense
                # if self.args.debug:
                #     import matplotlib.pyplot as plt
                #     fig, axs = plt.subplots(figsize=(6, 4))
                #     axs.triplot(ret['pred_V'][:,0].detach().cpu().numpy(), ret['pred_V'][:,1].detach().cpu().numpy(), ret['T'], linewidth=0.5)
                #     plt.axis('off')
                #     plt.savefig(f"scratch/{source.source_ind}_fuv_pred.png")
                #     plt.close(fig)
                #     plt.cla()

            # if self.args.test:
            #     ret['pred_J_R'] = poisson_J_restricted.detach()
            #     ret['target_J_R'] = GT_J_restricted.detach()

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
        vertices = batch_parts["source_V"].detach().cpu().numpy()
        faces = batch_parts["ogT"]
        # mesh = Mesh(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"])

        # 3d positions as uv colors
        fvs = vertices[faces] # F x 3 x 3
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
            axs.set_title(f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx}: SP Weights")
            axs.hist(batch_parts['weights'], bins=20)
            plt.savefig(os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png"))
            plt.close(fig)
            plt.cla()
            self.logger.log_image(key=f'{pref} weights', images=[os.path.join(save_path, f"weights_epoch_{self.current_epoch:05}_batch{batch_idx}.png")],
                                    step=self.current_epoch)

        ### Texture images
        # Save rendered images for debugging
        if len(self.args.visualloss) > 0 or self.args.showtexture:
            import matplotlib.pyplot as plt

            # Save rendered images
            for texturename in source.texturenames:
                if texturename not in self.textureimg:
                    continue

                if f'{texturename}_predrenders' not in lossdict[0]:
                    continue

                images = lossdict[0][f'{texturename}_predrenders']
                num_views = len(images)

                # If debugging, then make these big
                if self.args.debug:
                    fig, axs = plt.subplots(int(np.ceil(num_views/5)), 5, figsize=(30,5))
                else:
                    fig, axs = plt.subplots(int(np.ceil(num_views/5)), 5)

                for nview in range(num_views):
                    j = nview % 5
                    if num_views > 5:
                        i = nview // 5
                        axs[i,j].imshow(images[nview])
                        axs[i,j].axis('off')
                    else:
                        axs[j].imshow(images[nview])
                        axs[j].axis('off')

                plt.axis('off')
                fig.suptitle(f"{pref} Epoch {self.current_epoch} Batch {batch_idx} {texturename}")
                plt.savefig(os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_texture.png"))
                plt.close(fig)
                plt.cla()

            # Log the plotted imgs
            images = [os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_texture.png") for texturename in source.texturenames \
                if texturename in self.textureimg]

            # Remove images that don't exist
            images = [img for img in images if os.path.exists(img)]

            if len(images) > 0:
                self.logger.log_image(key=f'{pref} textures', images=images, step=self.current_epoch)

            ## Plot UVs overlaid on the texture img
            from results_saving_scripts.plot_uv import plot_texture

            ## Tile
            imgv = batch_parts['pred_V'].squeeze().detach().cpu().numpy()

            # Testing out of bounds behavior
            if self.args.debug:
                imgv *= 2

            # Compute the extents based on the UV range
            # NOTE: Need UV extents to be at least 0 to show the UV translation accurately
            xmax = max(np.max(np.ceil(imgv[:,0])), 0)
            xmin = min(np.min(np.floor(imgv[:,0])), 0)
            ymax = max(np.max(np.ceil(imgv[:,1])), 0)
            ymin = min(np.min(np.floor(imgv[:,1])), 0)

            # Number of dups needed is simply the max of the extent range
            dups = int(np.ceil(max(xmax - xmin, ymax - ymin)))

            # Number of tiles is the max dups for any coordinate
            ntiles = dups

            # NOTE: We still need to normalize UVs for kaolin texture sampling to work
            # Normalize the UVs
            # with torch.no_grad():
            #     centroid = torch.mean(imgv.reshape(-1, 2), dim=0)
            #     tmpuv = imgv - centroid
            #     r = 2 * torch.max(torch.linalg.norm(tmpuv, dim=1))

            # imgv = imgv - centroid
            # imgv /= r
            # imgv = imgv + 0.5
            # imgv = imgv.numpy()

            # Save rendered images
            for texturename in source.texturenames:
                if texturename not in self.textureimg:
                    continue

                # # Find index of texture paths which correspond with the texture name
                # for i in range(len(self.texturenames)):
                #     if self.texturenames[i] == texturename:
                #         imgi = i
                #         break

                # imgpath = self.args.textureimg[imgi]

                from PIL import Image
                import torchvision

                textureimg = self.textureimg[texturename].cpu().numpy()
                tiletex = np.tile(textureimg, (1, ntiles, ntiles))

                plot_texture(os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_uv.png"),
                             imgv, batch_parts["T"], np.transpose(tiletex, (1,2,0)),
                             name=f"Epoch {self.current_epoch}: {texturename} UVs",
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,)

                # Save the latest UV overlays
                import shutil
                shutil.copy(os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_uv.png"),
                            os.path.join(source_path, f"latest_{texturename}_uv.png"))

            ## TODO: If SDS, then also save the target denoised images from diffusion
            if 'sds' in self.args.visualloss:
                pass

            ## Save GT images if available
            if 'render' in self.args.visualloss:
                for texturename in source.texturenames:
                    if texturename not in self.textureimg:
                        continue

                    images = lossdict[0][f'{texturename}_gtrenders']
                    num_views = len(images)
                    fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
                    for nview in range(num_views):
                        j = nview % 5
                        if nview > 5:
                            i = nview // 5
                            axs[i,j].imshow(images[nview])
                            axs[i,j].axis('off')
                        else:
                            axs[j].imshow(images[nview])
                            axs[j].axis('off')
                    plt.axis('off')
                    fig.suptitle(f"{pref} Epoch {self.current_epoch} Batch {batch_idx} {texturename}")
                    plt.savefig(os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_gttexture.png"))
                    plt.close(fig)
                    plt.cla()

                images = [os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}_gttexture.png") \
                    for texturename in source.texturenames if texturename in self.textureimg]
                self.logger.log_image(key=f'{pref} gt textures', images=images, step=self.current_epoch)

            ### Save 360 views of the source mesh + gif
            import torchvision
            import torch
            gifs = []

            elev = torch.zeros(20).float().to(self.device)
            azim = torch.linspace(0, 2*np.pi, 20).float().to(self.device)
            for texturename in source.texturenames:
                if texturename not in self.textureimg:
                    continue

                tex = self.textureimg[texturename].float().to(self.device)

                if self.args.rasterizer == "kaolin":
                    self.renderer.device = self.device
                    self.renderer.camera_projection = self.renderer.camera_projection.to(self.device)
                    self.renderer.lights = self.renderer.lights.to(self.device)
                    self.renderer.background = self.renderer.background.to(self.device)

                    if self.args.up == 'x':
                        up = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
                    elif self.args.up == 'y':
                        up = torch.tensor([0.0, 1.0, 0.0]).to(self.device)
                    else:
                        up = torch.tensor([0.0, 0.0, 1.0]).to(self.device)

                    pred_render, pred_mask = self.renderer.render_texture(torch.from_numpy(vertices).to(self.device).float(),
                                                                          torch.from_numpy(faces).to(self.device).long(),
                                                                          batch_parts["pred_V"].reshape(-1, 3, 2).float(),
                                                                            tex, elev=elev, azim=azim, white_background=True,
                                                                            up = up, radius=2.7, mod=True, )

                # Save renders in a gif
                imgs = [torchvision.transforms.functional.to_pil_image(pred_render[i].cpu().detach()) for i in range(len(pred_render))]

                # Resize images
                fp_out = os.path.join(save_path, f"{self.current_epoch:05}_{pref}_{source.source_ind}_{texturename}.gif")

                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=0)

                gifs.append(fp_out)

                # Copy the latest gifs over to the parent
                import shutil
                shutil.copy(fp_out, os.path.join(source_path, f"latest_{texturename}.gif"))

                ##### Validation checkpointing: check either CLIP loss or fixed timestep SDS loss w/ views
                if validation and texturename in source.texturedescrip.keys():
                    texturedescrip = source.texturedescrip[texturename]

                    # CLIP loss
                    if self.args.clipcheckpointing:
                        import clip

                        pred_render = pred_render[:,:3]

                        # Make sure the model is in float
                        self.clip_model = self.clip_model.type(torch.float32)

                        with torch.no_grad():
                            clip_renders = self.normalize_transform(pred_render)
                            prompt_token = clip.tokenize([texturedescrip]).to(self.device)
                            encoded_text = self.clip_model.encode_text(prompt_token)
                            encoded_renders = self.clip_model.encode_image(clip_renders)
                            cliploss = (1 - torch.cosine_similarity(encoded_text, encoded_renders, dim=-1)).detach().cpu().numpy()
                        cliploss = np.mean(cliploss)

                        if cliploss < self.best_cliploss[texturename]:
                            import dill as pickle

                            self.best_cliploss[texturename] = cliploss
                            self.best_cliploss_epoch[texturename] = self.current_epoch

                            # Save the best
                            with open(os.path.join(source_path, f"best_cliploss.pkl"), 'wb') as f:
                                pickle.dump(self.best_cliploss, f)

                            with open(os.path.join(source_path, f"best_cliploss_epoch.pkl"), 'wb') as f:
                                pickle.dump(self.best_cliploss_epoch, f)

                    if self.args.sdscheckpointing:
                        # Fixed timestep SDS loss
                        text_z, text_z_neg = self.diffusion.encode_prompt(texturedescrip)
                        timesteps = [.02, .25, .5, .75, .98]

                        totsds = []

                        for t in timesteps:
                            sds = self.diffusion(pred_render[:,:3], mask_img = pred_mask, prompt_embeds=text_z, tratio=t, stage='I')
                            sdsloss = sds['loss_sds']
                            totsds.append(sdsloss.item())

                            # Stage 2 if cascaded
                            if self.args.sdsloss == 'csd':
                                sds = self.diffusion(pred_render[:,:3], mask_img = pred_mask, prompt_embeds=text_z, tratio=t, stage='II')
                                totsds.append(sds['loss_sds'].item())
                        totsds = np.mean(totsds)

                        if totsds < self.best_sdsloss[texturename]:
                            import dill as pickle

                            self.best_sdsloss[texturename] = totsds
                            self.best_sdsloss_epoch[texturename] = self.current_epoch

                            # Save the best
                            with open(os.path.join(source_path, f"best_sdsloss.pkl"), 'wb') as f:
                                pickle.dump(self.best_sdsloss, f)

                            with open(os.path.join(source_path, f"best_sdsloss_epoch.pkl"), 'wb') as f:
                                pickle.dump(self.best_sdsloss_epoch, f)

            # Log all the gifs
            self.logger.log_image(key=f'{pref} textures', images=gifs, step=self.current_epoch)

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
                uv = source.prefuv.reshape(-1, 2)
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
                export_views(source.cutvs, source.cutfs, save_path, filename=f"{pref}_boundary_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"Initial Mesh Boundary Batch {batch_idx}", cylinders=totboundaries,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

        ### Compute cut edges based on stitching loss (on original topology and soup topology) ###
        edgeseparation = None
        if 'pred_V' in batch_parts.keys() and not self.args.gtjacobianloss:
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
        else:
            soup_cutedges_stitch = []
            soup_cutedgecolors_stitch = []

        # Compute flips
        if 'pred_V' in batch_parts.keys() and not self.args.gtjacobianloss:
            from utils import get_flipped_triangles
            flipped = get_flipped_triangles(batch_parts["pred_V"].squeeze().detach().cpu().numpy(), batch_parts['T'].squeeze())
            flipvals = np.zeros(len(batch_parts['T'].squeeze()))
            flipvals[flipped] = 1
            lossdict[0]['fliploss'] = flipvals

        # Compute distortion if not already in loss
        if 'pred_V' in batch_parts.keys():
            if 'distortionloss' not in lossdict[0].keys() and not self.args.gtjacobianloss:
                from source_njf.losses import arap, symmetricdirichlet
                # distortionenergy = arap(batch_parts["source_V"], torch.from_numpy(batch_parts["ogT"]).to(self.device),
                #                         batch_parts["pred_V"],
                #                         paramtris = batch_parts["pred_V"].reshape(-1, 3, 2),
                #                         device=self.device,
                #                         renormalize=False,
                #                         return_face_energy=True, timeit=False)
                distortionenergy = symmetricdirichlet(batch_parts["source_V"], torch.from_numpy(batch_parts["ogT"]).to(self.device),
                                                    batch_parts['pred_J_poiss'].squeeze())
                lossdict[0]['distortionloss'] = distortionenergy.detach().cpu().numpy()

                if validation:
                    self.log("val distortionloss", np.mean(distortionenergy.detach().cpu().numpy()), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

            if len(batch_parts["pred_V"].shape) == 4:
                for idx in range(len(batch_parts["pred_V"])):
                    plot_uv(save_path, f"{pref} epoch {self.current_epoch:05} id {idx:05} batch {batch_idx}", batch_parts["pred_V"][idx].squeeze().detach().cpu().numpy(),
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
            images = [os.path.join(source.source_dir, "..", "..", f"{source.source_ind}_3dpos.png")] + \
                    [os.path.join(save_path, f"{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + \
                    [os.path.join(save_path, f"{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}_edges.png")] + \
                        [os.path.join(save_path, f"{key}_{pref}_epoch_{self.current_epoch:05}_batch_{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

            if self.args.init:
                images = [os.path.join(save_path, f"{pref}_{self.args.init}_init_epoch_{self.current_epoch:05}_batch_{batch_idx}.png")] + images

            # Log GT UVs
            if self.args.gtuvloss or self.args.gtnetworkloss:
                images = [os.path.join(source.source_dir, "..", "..", "gtuv.png")] + images

            # Filter out all renders that dont exist
            images = [imgpath for imgpath in images if os.path.exists(imgpath)]

            # Save the latest stuff
            import shutil
            import re
            for image in images:
                fname = os.path.basename(image)
                research = re.search(r"(.*)_epoch_\d+_batch_\d+(.*)\.png", fname)
                if research is None:
                    continue
                newname = f"latest_{research.group(1)}{research.group(2)}.png"
                shutil.copy(image,
                            os.path.join(source_path, newname))

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
                    plt.savefig(os.path.join(save_path, f"preduv_{fgroup}_{self.current_epoch:05}_batch_{batch_idx}.png"))
                    plt.close(fig)
                    plt.cla()

                # Log
                images = []
                for fgroup in np.unique(fgroups):
                    images.append(os.path.join(source.source_dir, "..", "..", f"gtuv_{fgroup}.png"))
                    images.append(os.path.join(save_path, f"preduv_{fgroup}_{self.current_epoch:05}_batch_{batch_idx}.png"))

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
        if self.args.hardpoisson:
            if 'pred_V' not in batch_parts.keys():
                predV = None
            else:
                predV = batch_parts['pred_V']

            if 'weights' not in batch_parts.keys():
                weights = None
            else:
                weights = batch_parts['weights']

            hp_uv, hp_cutedges, hp_cutpairs, hp_distortion, hp_cutlen, hp_vertices, hp_faces, hp_fidx = self.hardpoisson(source, predV,
                                                                     batch_parts['pred_J'][0, :, :2, :], weights,
                                                                     cuttype=self.args.hardpoisson, threshold=self.args.cuteps,
                                                                     edgeseparation=edgeseparation)

            # Log the hardpoisson energies
            self.log('hp cutlen', hp_cutlen, logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

            # Keep track of the OG face losses
            face_losses = ['gtjloss', 'invjloss', 'normalloss']

            ## Check for multiple connected components
            if isinstance(hp_uv, list):
                self.log('hp distortion', np.mean(np.concatenate(hp_distortion)), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

                images = []
                for i in range(len(hp_uv)):
                    # Visualize
                    import matplotlib.pyplot as plt

                    losses = {'distortionloss': hp_distortion[i]}

                    for key, val in lossdict[0].items():
                        if key in face_losses:
                            losses[key] = val[hp_fidx[i]]

                    plot_uv(save_path, f"{pref} component {i} hard poisson epoch {self.current_epoch:05} batch {batch_idx} cut length {np.sum(hp_cutlen):04f}", hp_uv[i],
                                hp_faces[i], losses=losses,
                                edge_cmap = plt.get_cmap("Reds"))

                    images += [os.path.join(save_path, f"{pref}_component_{i}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")] + \
                            [os.path.join(save_path, f"distortionloss_{pref}_component_{i}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")]

                    ## HACK: Manual truncate hard poisson list after 10 components
                    if i >= 10:
                        break

                ### Composite UVs
                losses = {'distortionloss': np.concatenate(hp_distortion)}

                hp_uv_tot = []
                hp_faces_tot = []
                totvcount = 0
                for i in range(len(hp_uv)):
                    hp_uv_tot.append(hp_uv[i])
                    hp_faces_tot.append(hp_faces[i] + totvcount)
                    totvcount += hp_uv[i].shape[0]
                hp_uv_tot = np.concatenate(hp_uv_tot, axis=0)
                hp_faces_tot = np.concatenate(hp_faces_tot, axis=0)

                plot_uv(save_path, f"{pref} hard poisson epoch {self.current_epoch:05} batch {batch_idx} cut length {np.sum(hp_cutlen):04f}", hp_uv_tot,
                            hp_faces_tot, losses=losses,
                            edge_cmap = plt.get_cmap("Reds"))

                images = [os.path.join(save_path, f"{pref}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")] + \
                        [os.path.join(save_path, f"distortionloss_{pref}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")]

            else:
                self.log('hp distortion', np.mean(hp_distortion), logger=True, prog_bar=False, batch_size=1, on_epoch=True, on_step=False)

                # Visualize
                import matplotlib.pyplot as plt

                losses = {'distortionloss': hp_distortion}

                for key, val in lossdict[0].items():
                    if key in face_losses:
                        losses[key] = val

                plot_uv(save_path, f"{pref} hard poisson epoch {self.current_epoch:05} batch {batch_idx} cut length {np.sum(hp_cutlen):04f}", hp_uv,
                            hp_faces, losses=losses,
                            edge_cmap = plt.get_cmap("Reds"))

                images = [os.path.join(save_path, f"{pref}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")] + \
                        [os.path.join(save_path, f"distortionloss_{pref}_hard_poisson_epoch_{self.current_epoch:05}_batch_{batch_idx}_cut_length_{np.sum(hp_cutlen):04f}.png")]

            # Plot edge cuts
            ogvs = batch_parts["source_V"].detach().cpu().numpy()
            cylinderpos = ogvs[hp_cutpairs]

            # Stitch cut
            if len(cylinderpos) == 0:
                cylinderpos = None

            export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_mesh_hardpoisson_{self.current_epoch:05}_batch{batch_idx}.png",
                        plotname=f"({pref}) Epoch {self.current_epoch:05} Batch {batch_idx} Total cut length: {hp_cutlen:0.5f}", cylinders=cylinderpos,
                        outline_width=0.01, cmap = plt.get_cmap('Reds'),
                        device="cpu", n_sample=30, width=200, height=200,
                        vmin=0, vmax=1, shading=False)
            images += [os.path.join(save_path, f"{pref}_mesh_hardpoisson_{self.current_epoch:05}_batch{batch_idx}.png")]

            self.logger.log_image(key=f'{pref} hard poisson', images=images, step=self.current_epoch)

        ### Losses on 3D surfaces
        ### NOTE: mesh is original mesh topology (not soup)
        ogvs = batch_parts["source_V"].detach().cpu().numpy()
        ogfs = batch_parts["ogT"]

        images = []
        if 'weights' in batch_parts.keys():
            weights = batch_parts['weights']
            cylinderpos = ogvs[source.ogedge_vpairs_nobound.detach().cpu().numpy()]
            cylindervals = np.stack([weights, weights], axis=1) # E x 2

            export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                        plotname=f"Edge Weights Epoch {self.current_epoch:05} Batch {batch_idx}", cylinders=cylinderpos,
                        cylinder_scalars=cylindervals,
                        outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                        device="cpu", n_sample=30, width=200, height=200,
                        vmin=0, vmax=1, shading=False)
            images += [os.path.join(save_path, f"{pref}_weights_mesh_{self.current_epoch:05}_batch{batch_idx}.png")]

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
                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}", outline_width=0.01,
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

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals, outline_width=0.01,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=self.args.edgecut_weight, shading=False,
                                subcylinders = subcylinders)

                elif key == "gtweightloss":
                    from collections import defaultdict
                    cylinderpos = ogvs[source.ogedge_vpairs_nobound.detach().cpu().numpy()]
                    cylindervals = np.stack([val, val], axis=1) # E x 2

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
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

                    export_views(ogvs, batch_parts["ogT"], save_path, filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals, outline_width=0.01,
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=0.5, shading=False)

                elif key in ["distortionloss", "gtjloss", "normalloss"]:
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "gtuvloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}",
                                    vcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "invjloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                elif key == "fliploss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=0.6, shading=False)
                elif key == "intersectionloss":
                    export_views(batch_parts["source_V"].detach().cpu().numpy(), batch_parts["ogT"], save_path,
                                    filename=f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png",
                                    plotname=f"{pref} Epoch {self.current_epoch:05} Batch {batch_idx} Avg {key}: {np.mean(val):0.4f}",
                                    fcolor_vals=val, device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)
                else:
                    continue

        # Log together: 3D surface losses + initial tutte cut
        images = [os.path.join(save_path, f"{pref}_{key}_mesh_{self.current_epoch:05}_batch{batch_idx}.png") for key in lossdict[0].keys() if "loss" in key]

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


def main(gen, args):
    pl.seed_everything(48, workers=True)

    # Save directory
    save_path = os.path.join(args.outputdir, args.expname)
    if args.overwrite and os.path.exists(save_path) and not args.continuetrain and not args.test:
        from utils import clear_directory
        clear_directory(save_path)
    Path(save_path).mkdir(exist_ok=True, parents=True)

    if not args.compute_human_data_on_the_fly:
        ### DEFAULT GOES HERE ###
        with open(os.path.join(args.root_dir_train, args.data_file)) as file:
            data = json.load(file)
            pairs_train = data['pairs']
    else:
        if args.experiment_type == "REGISTER_TEMPLATE":
            pairs_train = [(f"ren_template", f"{(2*i):08d}") for i in range(args.size_train)]
        elif  args.experiment_type == "TPOSE":
            pairs_train = [(f"{(2*i):08d}", f"{(2*i+1):08d}") for i in range(args.size_train)]
    print("TRAIN :", len(pairs_train))

    model = None
    LOADING_CHECKPOINT = isinstance(gen, str)
    if LOADING_CHECKPOINT:
        model = load_network_from_checkpoint(gen, args)
        LOADING_CHECKPOINT = gen
        gen = model.encoder

    if args.split_train_set:
        train_max_ind = math.ceil(len(pairs_train) * args.train_percentage / 100)
        train_pairs = pairs_train[:train_max_ind]
        valid_pairs = pairs_train[train_max_ind:min(len(pairs_train), train_max_ind + 10000)]
        if args.test:
            if args.test_set == 'valid':
                print('++++++++++++++++++++++ TESTING ON VALIDATION PART +++++++++++++++++++++++')
                train_pairs = valid_pairs
            if args.test_set == 'train':
                print('++++++++++++++++++++++ TESTING ON TRAINING PART +++++++++++++++++++++++')
            if args.test_set == 'all':
                print('++++++++++++++++++++++ TESTING ON ENTIRE DATASET +++++++++++++++++++++++')
                train_pairs = pairs_train

    else:
        ### DEFAULT GOES HERE ###
        with open(os.path.join(args.root_dir_test, args.test_file)) as file:
            data = json.load(file)
            pairs_test = data['pairs']

        print("TEST :", len(pairs_test))
        valid_pairs = pairs_test
        train_pairs = pairs_train

    id = None
    if args.continuetrain:
        import re
        if os.path.exists(os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')):
            for idfile in os.listdir(os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')):
                if idfile.endswith(".wandb"):
                    result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                    if result is not None:
                        id = result.group(1)
                        break
        else:
            print(f"Warning: No wandb record found in {os.path.join(args.outputdir, args.expname, 'wandb', 'latest-run')}!. Starting log from scratch...")

    # TODO: ENABLE WHEN WANDB STORAGE IS FIXED
    logger = WandbLogger(project=args.projectname, name=args.expname, save_dir=os.path.join(args.outputdir, args.expname), log_model=False,
                         offline=args.debug, resume='must' if args.continuetrain and id is not None else 'allow', id = id,
                         mode='disabled')

    # if args.gpu_strategy:
    #     if os.name != 'nt':  # no support for windows because of gloo
    #         if args.gpu_strategy == 'ddp':
    #             plugins = pl.plugins.training_type.DDPPlugin(find_unused_parameters=False)
    #         elif args.gpu_strategy == 'ddp_spawn':
    #             plugins = pl.plugins.training_type.DDPSpawnPlugin(find_unused_parameters=False)
    #

    checkpoint_callback = ModelCheckpoint(save_on_train_epoch_end=True, dirpath=os.path.join(save_path, "ckpt"), every_n_epochs=1)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    ################################ TRAINER #############################
    trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=args.precision,
                         max_epochs=args.epochs, sync_batchnorm=args.n_devices != 1,
                         check_val_every_n_epoch=args.val_interval,
                         logger=logger,
                         plugins=[SLURMEnvironment(requeue_signal=SIGUSR1)] if not args.debug else None,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         num_sanity_val_steps=0,
                         limit_val_batches = 0, # NOTE: This turns off validation_step
                         enable_model_summary=False,
                         enable_progress_bar=True,
                         num_nodes=1,
                         gradient_clip_val=args.gradclip,
                         deterministic= args.deterministic,
                         callbacks=[checkpoint_callback,lr_monitor],
                         )
    # trainer.replace_sampler_ddp = False
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

    train_dataset = DeformationDataset(train_pairs, gen.get_keys_to_load(True),
                                       gen.get_keys_to_load(False), use_dtype, train=True, args=args)

    # Initialize the highest weights dim here
    # if args.softpoisson:
    #     train_dataset.get_weights_dim()
    #     print(f"\n========== Soft poisson training: weights dim is {train_dataset.weightsdim} ========== \n")

    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, pin_memory = (args.unpin_memory is None),
                              shuffle=(args.test is None), num_workers=args.workers, persistent_workers=args.workers > 0)

    if args.no_validation or args.test:
        valid_loader = None
    else:
        valid_dataset = DeformationDataset(valid_pairs, gen.get_keys_to_load(True),
                                        gen.get_keys_to_load(False), use_dtype, train=False, args=args)

        valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                  shuffle=False, num_workers=0, persistent_workers=0)

    # ================ #
    # model
    # ================ #
    gen.type(use_dtype)
    if model is None:
        assert (isinstance(gen, DeformationEncoder.DeformationEncoder))
        model = MyNet(gen.get_code_length(train_dataset), encoder = gen, point_dim=train_dataset.get_point_dim(), args=args,
                      face_dim = args.facedim, n_edges=train_dataset.get_edges() if args.arch in ['fullconv','directedge'] else None)

    # NOTE: Network not initializing with correct device!!!
    if has_gpu == "gpu":
        device = torch.device("cuda:0")
        model.to(device)
        model.lossfcn.device = device

        if hasattr(model, 'diffusion'):
            model.diffusion.device = device
            model.diffusion.stage_I_alphas = model.diffusion.stage_I_alphas.to(device)
            model.diffusion.stage_I_pipe = model.diffusion.stage_I_pipe.to(device)

            if hasattr(model.diffusion, "stage_II_pipe"):
                model.diffusion.stage_II_alphas = model.diffusion.stage_II_alphas.to(device)
                model.diffusion.stage_II_pipe = model.diffusion.stage_II_pipe.to(device)

        if hasattr(model, 'diffusion_cnet'):
            model.diffusion_cnet.device = device
            model.diffusion_cnet.pipe = model.diffusion_cnet.pipe.to(device)
            model.diffusion_cnet.alphas = model.diffusion_cnet.alphas.to(device)

        if args.clipcheckpointing:
            model.clip_model = model.clip_model.to(device)

    else:
        model.to(torch.device("cpu"))
        model.lossfcn.device = torch.device("cpu")

    model.type(use_dtype)
    model.lr = args.lr

    # CLIP doesn't work with float64 dtype
    if use_dtype == torch.double and args.clipcheckpointing:
        model.clip_model = model.clip_model.type(torch.float32)

    # trainer.tune(model)
    if args.overfit_one_batch:
        print("=======OVERFITTING========")
        # Going to attempt something a bit risky because it is just too slow. The goal is to load a batch once and for all and overfit it.
        overfitting_batch = next(iter(train_loader))

        class LocalDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.batch = batch

            def __getitem__(self, idx):
                return self.batch

            def __len__(self):
                return 1

        local_dataset = LocalDataset(overfitting_batch)
        train_loader = DataLoader(local_dataset, batch_size=1, collate_fn=custom_collate, pin_memory=(args.unpin_memory is None),
                                  num_workers=0)

        trainer = pl.Trainer(accelerator=has_gpu, devices=args.n_devices, precision=32, max_epochs=args.epochs,
                             overfit_batches=1)
        trainer.fit(model, train_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)
        return

    if args.test:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& TEST &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        trainer.test(model, train_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)
        return

    # If loss args doesn't have stitchrelax, then add it
    if not hasattr(model.lossfcn.args, "stitchrelax"):
        model.lossfcn.args.stitchrelax = True

    if not hasattr(model.lossfcn.args, "stitchlossweight"):
        model.lossfcn.args.stitchlossweight = 1

    trainer.fit(model, train_loader, valid_loader, ckpt_path=LOADING_CHECKPOINT if LOADING_CHECKPOINT else None)

    # Save UVs
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    for idx, data in enumerate(train_loader):
        devdata = (data[0].to(model.device), data[1].to(model.device))
        ret = model.my_step(devdata, idx, validation=True)
        source, target = data
        sourcepath = source.source_dir
        np.save(os.path.join(sourcepath, f"latest_preduv_{idx}.npy"), ret['pred_V'].squeeze().detach().cpu().numpy())
        np.save(os.path.join(sourcepath, f"latest_predt_{idx}.npy"), ret['T'])

        # if args.no_poisson:
        #     np.save(os.path.join(sourcepath, f"latest_poissonuv.npy"), ret['poissonUV'].squeeze().detach().cpu().numpy())
        #     np.save(os.path.join(sourcepath, f"latest_poissont.npy"), ret['ogT']) # NOTE: poisson T uses original triangles!

    ### GENERATE GIFS
    pref = ""
    # if args.lossgradientstitching:
    #     pref = "gradstitch_"

    from PIL import Image
    import glob
    import re
    for batchi, batch in enumerate(train_loader):
        source, target = batch
        sourcename = os.path.basename(source.source_dir)
        vispath = os.path.join(save_path, "renders", sourcename)

        for pref in ['train', 'val']:
            ## Default UV gif
            fp_in = f"{vispath}/frames/{pref}_epoch_*_batch_{batchi}.png"
            fp_out = f"{vispath}/{pref}_{batchi}.gif"
            imgs = [Image.open(f) for f in sorted(glob.glob(fp_in)) if re.search(r'.*(\d+)\.png', f)]

            if len(imgs) > 0:
                # Resize images
                basewidth = 400
                wpercent = basewidth/imgs[0].size[0]
                newheight = int(wpercent * imgs[0].size[1])
                imgs = [img.resize((basewidth, newheight)) for img in imgs]

                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=0)

            ### Texture UVs
            for texturename in source.texturenames:
                fp_in = f"{vispath}/frames/*_{pref}_{source.source_ind}_{texturename}_uv.png"
                fp_out = f"{vispath}/{pref}_{source.source_ind}_{texturename}.gif"
                imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                if len(imgs) == 0:
                    continue

                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=0)

            ## Individual losses
            lossnames = model.lossfcn.lossnames
            # if args.lossgradientstitching:
            #     lossnames.append('stitchdistortionloss')

            uv_keys = [f"{vispath}/{pref}_{batchi}.gif"]
            mesh_keys = []
            for key in lossnames:
                if "loss" in key:
                    # Embedding viz
                    fp_in = f"{vispath}/frames/{key}_{pref}_epoch_*_batch_{batchi}.png"
                    fp_out = f"{vispath}/{pref}_{key}_batch{batchi}.gif"
                    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                    if len(imgs) == 0:
                        continue

                    # Resize images
                    basewidth = 400
                    wpercent = basewidth/imgs[0].size[0]
                    newheight = int(wpercent * imgs[0].size[1])
                    imgs = [img.resize((basewidth, newheight)) for img in imgs]

                    imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                            save_all=True, duration=100, loop=0, disposal=0)

                    uv_keys.append(fp_out)

                    # Mesh viz
                    fp_in = f"{vispath}/frames/{pref}_{key}_mesh_*_batch{batchi}.png"
                    fp_out = f"{vispath}/{pref}_{key}_mesh_batch{batchi}.gif"
                    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                    # Resize images
                    basewidth = 1000
                    wpercent = basewidth/imgs[0].size[0]
                    newheight = int(wpercent * imgs[0].size[1])
                    imgs = [img.resize((basewidth, newheight)) for img in imgs]

                    imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                            save_all=True, duration=100, loop=0, disposal=0)
                    mesh_keys.append(fp_out)

            if len(uv_keys) > 1:
                model.logger.log_image(key=f"{pref} uv gif", images=uv_keys)
            if len(mesh_keys) > 1:
                model.logger.log_image(key=f"{pref} mesh gif", images=mesh_keys)

            # Hard poisson
            # TODO: gif for hard poisson cuts
            # TODO: gif for the compiled hard poisson UVs
            if args.hardpoisson:
                fp_in = f"{vispath}/frames/distortionloss_{pref}_hard_poisson_epoch_*_batch_{batchi}_*.png"
                fp_out = f"{vispath}/{pref}_hardpoisson_{batchi}.gif"
                imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                # Resize images
                basewidth = 400
                wpercent = basewidth/imgs[0].size[0]
                newheight = int(wpercent * imgs[0].size[1])
                imgs = [img.resize((basewidth, newheight)) for img in imgs]
                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=0)

                ## Hard poisson mesh cuts
                fp_in = f"{vispath}/frames/{pref}_mesh_hardpoisson_*_batch{batchi}.png"
                fp_out = f"{vispath}/{pref}_mesh_hardpoisson_{batchi}.gif"
                imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

                # Resize images
                basewidth = 400
                wpercent = basewidth/imgs[0].size[0]
                newheight = int(wpercent * imgs[0].size[1])
                imgs = [img.resize((basewidth, newheight)) for img in imgs]
                imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
                        save_all=True, duration=100, loop=0, disposal=0)

                model.logger.log_image(key=f"hard poisson gif", images=[f"{vispath}/{pref}_hardpoisson_{batchi}.gif",
                                                                        f"{vispath}/{pref}_mesh_hardpoisson_{batchi}.gif"])
    # ================ #
