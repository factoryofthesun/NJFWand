import random
from re import I
import sys

from torch.utils.data import Dataset
import numpy as np
import os
import glob
from SourceMesh import SourceMesh
import time
from os.path import join
from pathlib import Path
from MeshProcessor import WaveKernelSignatureError
import logging
logging.basicConfig(level=logging.DEBUG, filename='./exception.log')

class UVDataset(Dataset):
    '''
    Main dataset. Each sample in it is a source <---> target pair. Note this dataset return a custom Batch class object instead of a
    tensor, which is already batched.
    '''
    def num_meshes(self):
        return len(self.file_names)

    def num_pairs(self):
        return self.len_pairs

    def __init__(self, objpaths, source_keys, ttype, args, train=False, cpuonly=False):
        '''
        :param objpaths: list of paths to .obj files
        :param max_source_batch: max batch_size. Note batches cannot have more than one unique source mesh.
        :param batch_class: which class to use to create batch objects
        '''
        self.cpuonly=cpuonly
        self.ttype = ttype
        self.train = train

        self.len = len(objpaths)

        # NOTE: If ninit is a positive integer, then we will cache n copies of the source inits
        if args.ninit > 0:
            self.objpaths = objpaths * args.ninit
        else:
            self.objpaths = objpaths
        self.source_keys = source_keys
        self.args = args
        self.directory = self.args.root_dir_train if self.train else self.args.root_dir_test
        self.source = None

    def __len__(self):
        return len(self.objpaths)

    def get_item_default(self,ind, verbose=False):
        source = None
        source_index = self.objpaths[ind]

        if Path(source_index).suffix  in [ '.obj' , '.off', '.ply']:
            # NOTE: Below caches the vertices/faces + creates the directory
            self.obj_to_npy(Path(join(self.directory, source_index)), ind)
            source_index = source_index[:-4]

        # ==================================================================
        # LOAD SOURCE
        # If unique, then there is only one source path
        if len(self.objpaths) == 1:
            source = self.source
        else:
            # TODO: ind should be tied to count per unique source not across all sources!
            source = SourceMesh(self.args, source_index, join(self.directory, 'cache', f"{source_index}_{ind}"), self.source_keys, self.ttype, use_wks = not self.args.no_wks,
                                random_centering=(self.train and self.args.random_centering),  cpuonly=self.cpuonly, init=self.args.init,
                                initjinput = self.args.initjinput, fft=self.args.fft, fftscale=self.args.fftscale,
                                debug=self.args.debug, top_k_eig=self.args.top_k_eig)
            new_init = None
            if self.args.ninit == -1:
                if not self.args.basistype:
                    new_init = True
                else:
                    new_init = self.args.basistype
            # TODO: NEED TO ADD CONDITION FOR WHETHER THE INDS BEEN INITIALIZED ALREADY
            elif self.args.ninit > 1:
                new_init = "constant"
            elif self.args.ninit == 1:
                new_init = True
            source.load(new_init= new_init)
            self.source = source

        return source

    def check_if_files_exist(self, paths):
        exist = True
        for path in paths:
            exist = (exist and path.is_file())
        return exist

    def obj_to_npy(self, path, ind):
        from meshing.io import PolygonSoup
        from meshing.mesh import Mesh
        # NOTE: All mesh data should be saved into 'cache'
        directory_name, basename = os.path.split(os.path.join(os.path.splitext(path)[0]))
        directory = os.path.join(directory_name, "cache", f"{basename}_{ind}")

        if not os.path.exists(join(directory , "vertices.npy")) and not os.path.exists(join(directory, "faces.npy")):
            os.makedirs(directory, exist_ok=True)
            soup = PolygonSoup.from_obj(path)
            mesh = Mesh(soup.vertices, soup.indices)

            # Normalize mesh
            mesh.normalize()

            np.save(join(directory , "vertices.npy"), mesh.vertices)
            np.save(join(directory , "faces.npy"), mesh.faces)

    def __getitem__(self,ind, verbose=False):
        start = time.time()
        if self.args.experiment_type == "DEFAULT":
            data_sample = self.get_item_default(ind)
        if verbose:
            print(f"DATALOADER : loaded sample in {time.time() - start}")

        return data_sample

    def get_point_dim(self):
        return self[0][0].get_point_dim()
