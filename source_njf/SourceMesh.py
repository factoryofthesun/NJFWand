import os

import numpy
import numpy as np
from numpy.random import default_rng
import torch
import igl
import MeshProcessor
WKS_DIM = MeshProcessor.WKS_DIM
WKS_FACTOR = 1000
import numpy as np
import sys
import random
import time
from utils import FourierFeatureTransform

class SourceMesh:
    '''
    datastructure for the source mesh to be mapped
    '''

    def __init__(self, args, source_ind, source_dir, extra_source_fields,
                 random_scale, ttype, use_wks=False, random_centering=False,
                cpuonly=False, init=False, fft=False, fft_dim=256, flatten=False,
                initjinput=False, debug=False, top_k_eig=50):
        self.args = args
        self.__use_wks = use_wks
        self.source_ind = source_ind
        # NOTE: This is the CACHE DIRECTORY
        self.source_dir = source_dir
        self.centroids_and_normals = None
        self.center_source = True
        self.poisson = None
        self.__source_global_translation_to_original = 0
        self.__extra_keys = extra_source_fields
        self.__loaded_data = {}
        self.__ttype = ttype
        self.__random_scale = random_scale
        self.random_centering = random_centering
        self.source_mesh_centroid = None
        self.mesh_processor = None
        self.cpuonly = cpuonly
        self.init = init
        self.initjinput = initjinput
        self.flatten = flatten
        self.debug = debug
        self.top_k_eig = top_k_eig

        self.fft = None
        if fft:
            # Compute input channels
            if self.__use_wks:
                n_input = 106
            else:
                n_input = 6

            # Initialize fourier features transform
            self.fft = FourierFeatureTransform(n_input, fft_dim)

    def get_vertices(self):
        return self.source_vertices

    def get_global_translation_to_original(self):
        return self.__source_global_translation_to_original

    def vertices_from_jacobians(self, d):
        return self.poisson.solve_poisson(d)

    def jacobians_from_vertices(self, v):
        return self.poisson.jacobians_from_vertices(v)

    def restrict_jacobians(self, J):
        return self.poisson.restrict_jacobians(J)

    def get_loaded_data(self, key: str):

        return self.__loaded_data.get(key)

    def get_source_triangles(self):
        # if self.__source_triangles is None:
        #     self.__source_triangles = np.load(os.path.join(self.source_dir, 'faces.npy'))
        return self.mesh_processor.get_faces()

    def to(self, device):
        self.poisson = self.poisson.to(device)
        self.centroids_and_normals = self.centroids_and_normals.to(device)
        for key in self.__loaded_data.keys():
            self.__loaded_data[key] = self.__loaded_data[key].to(device)
        return self

    ### PRECOMPUTATION HAPPENS HERE ###
    def __init_from_mesh_data(self, new_init=False):
        from meshing.mesh import Mesh
        from meshing.edit import EdgeCut
        from meshing.io import PolygonSoup

        assert self.mesh_processor is not None
        self.mesh_processor.prepare_differential_operators_for_use(self.__ttype) #call 1
        self.source_vertices = torch.from_numpy(self.mesh_processor.get_vertices()).type(
            self.__ttype)
        if self.__random_scale != 1:
            print("Diff ops and WKS need to be multiplied accordingly. Not implemented for now")
            sys.exit()
        self.source_vertices *= self.__random_scale

        bb = igl.bounding_box(self.source_vertices.numpy())[0]
        diag = igl.bounding_box_diagonal(self.source_vertices.numpy())

        self.source_mesh_centroid =  (bb[0] + bb[-1])/2
        if self.random_centering:
            # centering augmentation
            self.source_mesh_centroid =  self.source_mesh_centroid + [(2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2, (2*random.random() - 1)*diag*0.2]

        # Load input to NJF MLP
        centroids = self.mesh_processor.get_centroids()
        centroid_points_and_normals = centroids.points_and_normals
        if self.__use_wks:
            wks = WKS_FACTOR * centroids.wks
            centroid_points_and_normals = numpy.hstack((centroid_points_and_normals, wks))
        self.centroids_and_normals = torch.from_numpy(
            centroid_points_and_normals).type(self.__ttype)
        if self.center_source:
            c = self.source_mesh_centroid
            self.centroids_and_normals[:, 0:3] -= c
            self.source_vertices -= c
            self.__source_global_translation_to_original = c

        if self.fft is not None:
            self.centroids_and_normals = self.fft(self.centroids_and_normals)

        self.poisson = self.mesh_processor.diff_ops.poisson_solver

        ### Load ground truth jacobians if set
        if self.args.lossgt:
            gtdir = os.path.join(self.source_dir, "..", "..", "gtJ.npy")
            if not os.path.exists(gtdir):
                raise ValueError("No ground truth jacobians found at {}".format(gtdir))
            gtJ = np.load(gtdir).transpose(0, 2, 1)

            # if self.args.debug:
            #     # Sanity check poisson solve
            #     gtJ_sanity = np.concatenate([gtJ, np.zeros((gtJ.shape[0], 1, gtJ.shape[2]))], axis=1)[None,:,:,:]
            #     gtV = self.vertices_from_jacobians(torch.from_numpy(gtJ_sanity)).squeeze().detach().cpu().numpy()
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots()
            #     axs.triplot(gtV[:,0], gtV[:,1], self.get_source_triangles(), linewidth=0.5)
            #     plt.axis('off')
            #     plt.savefig(f"scratch/{self.source_ind}_gtuv.png")
            #     plt.close(fig)
            #     plt.cla()

        # First check if initialization cached
        # TODO: Isometric initialization with curriculum learning (only sample limited range of rotations)

        # Precompute Tutte if set
        if self.init == "tutte":
            if os.path.exists(os.path.join(self.source_dir, "tuttefuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tutteuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttej.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttetranslate.pt")) and \
                    not new_init:
                self.tuttefuv = torch.load(os.path.join(self.source_dir, "tuttefuv.pt"))
                self.tutteuv = torch.load(os.path.join(self.source_dir, "tutteuv.pt"))
                self.tuttej = torch.load(os.path.join(self.source_dir, "tuttej.pt"))
                self.tuttetranslate = torch.load(os.path.join(self.source_dir, "tuttetranslate.pt"))
            else:
                from utils import tutte_embedding, get_local_tris, generate_random_cuts

                vertices = self.source_vertices
                device = vertices.device
                faces = self.get_source_triangles()
                mesh = Mesh(vertices.detach().cpu().numpy(), faces)
                ogvs, ogfs, oges = mesh.export_soup()
                vertices = torch.from_numpy(ogvs).to(device)
                faces = torch.from_numpy(ogfs).long().to(device)
                fverts = vertices[faces]

                if new_init:
                    rng = default_rng()
                    n_cuts = rng.integers(self.args.min_cuts, self.args.max_cuts+1)
                    cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)

                    # Unit test: mesh is still connected
                    vs, fs, es = mesh.export_soup()
                    testsoup = PolygonSoup(vs, fs)
                    n_components = testsoup.nConnectedComponents()
                    assert n_components == 1, f"After cutting found {n_components} components!"

                    # Save new topology
                    self.cutvs = vs
                    self.cutfs = fs

                    # Only replace Tutte if no nan
                    newtutte = torch.from_numpy(tutte_embedding(vs, fs)).unsqueeze(0) # 1 x F x 2
                    set_new_tutte = False
                    if torch.all(~torch.isnan(newtutte)):
                        self.tutteuv = newtutte

                        # Convert Tutte to 3-dim
                        self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        meshprocessor = MeshProcessor.MeshProcessor.meshprocessor_from_array(vs, fs, self.source_dir, self._SourceMesh__ttype, cpuonly=self.cpuonly, load_wks_samples=self._SourceMesh__use_wks, load_wks_centroids=self._SourceMesh__use_wks)
                        meshprocessor.prepare_temporary_differential_operators(self._SourceMesh__ttype)
                        poissonsolver = meshprocessor.diff_ops.poisson_solver

                        # NOTE: We have NaNs here!!
                        self.tuttej = poissonsolver.jacobians_from_vertices(self.tutteuv) # F x 3 x 3

                        if torch.any(~torch.isfinite(self.tuttej)):
                            print("Tutte Jacobians have NaNs!")
                        else:
                            set_new_tutte = True
                    # Otherwise, just use the default Tutte
                    if not set_new_tutte:
                        self.tutteuv = torch.from_numpy(tutte_embedding(vertices.detach().cpu().numpy(), faces.detach().numpy())).unsqueeze(0) # 1 x V x 2

                        # Convert Tutte to 3-dim
                        self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        self.tuttej = self.jacobians_from_vertices(self.tutteuv) #  F x 3 x 3

                        # Reset cut topo to original topo
                        self.cutvs = vertices.detach().cpu().numpy()
                        self.cutfs = faces.detach().numpy()
                else:
                    self.tutteuv = torch.from_numpy(tutte_embedding(vertices.detach().cpu().numpy(), faces.detach().numpy())).unsqueeze(0) # 1 x V x 2

                    # Convert Tutte to 3-dim
                    self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                    # Get Jacobians
                    self.tuttej = self.jacobians_from_vertices(self.tutteuv) #  F x 3 x 3

                    # Reset cut topo to original topo
                    self.cutvs = vertices.detach().cpu().numpy()
                    self.cutfs = faces.detach().numpy()

                # DEBUG: make sure we can get back the original UVs up to global translation
                # NOTE: We compare triangle centroids bc face indexing gets messed up after cutting
                fverts = torch.from_numpy(ogvs[ogfs])
                # pred_V = torch.einsum("abc,acd->abd", (self.tuttej[0,:,:2,:], fverts)).transpose(1,2)
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.tuttej[0,:,:2,:].transpose(2,1)))

                if new_init and self.init == "tutte" and set_new_tutte:
                    checktutte = self.tutteuv[0,fs,:2]
                    self.tuttefuv = self.tutteuv[:,fs,:2] # B x F x 3 x 2
                else:
                    checktutte = self.tutteuv[0,faces,:2]
                    self.tuttefuv = self.tutteuv[:,faces,:2] # B x F x 3 x 2

                # diff = pred_V - checktutte
                # diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                # torch.testing.assert_allclose(diff.float(), torch.zeros(diff.shape), rtol=1e-4, atol=1e-4)

                ## Save the global translations
                self.tuttetranslate = (checktutte - pred_V)[:,:,:2]

                ## Compute ground truth if set
                if self.args.lossgt:
                    from numpy.linalg import pinv
                    invJ = pinv(self.tuttej.detach().cpu().numpy()[:,:,:2,:]) # 1 x F x 3 x 2
                    self.gtJ = torch.from_numpy(gtJ @ invJ) # 1 x F x 2 x 2
                    checkJ = self.gtJ @ self.tuttej[:,:,:2,:]
                    # np.testing.assert_allclose(checkJ, gtJ, atol=1e-5, rtol=1e-5)

                    torch.save(self.gtJ, os.path.join(self.source_dir, "tmpgtJ.pt"))

                # Cache everything
                torch.save(self.tuttefuv, os.path.join(self.source_dir, "tuttefuv.pt"))
                torch.save(self.tutteuv, os.path.join(self.source_dir, "tutteuv.pt"))
                torch.save(self.tuttej, os.path.join(self.source_dir, "tuttej.pt"))
                torch.save(self.tuttetranslate, os.path.join(self.source_dir, "tuttetranslate.pt"))

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['tuttefuv'] = self.tuttefuv
            self.__loaded_data['tutteuv'] = self.tutteuv
            self.__loaded_data['tuttej'] = self.tuttej
            self.__loaded_data['tuttetranslate'] = self.tuttetranslate
            if self.args.lossgt:
                self.__loaded_data['gtJ'] = self.gtJ

            if self.initjinput:
                self.centroids_and_normals = torch.cat([self.centroids_and_normals, self.tuttej.reshape(len(self.centroids_and_normals), -1)], dim=1)

            # self.__loaded_data['localj'] = self.localj

            # Debugging: plot the initial embedding
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(figsize=(6, 4))
            # # plot ours
            # axs.triplot(self.tutteuv[0,:,0], self.tutteuv[0,:,1], self.get_source_triangles(), linewidth=0.5)
            # plt.axis('off')
            # plt.savefig(f"scratch/{self.source_ind}.png")
            # plt.close(fig)
            # plt.cla()
        elif self.init == "slim":
            if os.path.exists(os.path.join(self.source_dir, "slimfuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimj.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimtranslate.pt")) and \
                    not new_init:
                self.slimfuv = torch.load(os.path.join(self.source_dir, "slimfuv.pt"))
                self.slimuv = torch.load(os.path.join(self.source_dir, "slimuv.pt"))
                self.slimj = torch.load(os.path.join(self.source_dir, "slimj.pt"))
                self.slimtranslate = torch.load(os.path.join(self.source_dir, "slimtranslate.pt"))
            else:
                from utils import SLIM, get_local_tris, generate_random_cuts

                vertices = self.source_vertices
                device = vertices.device
                faces = self.get_source_triangles()
                mesh = Mesh(vertices.detach().cpu().numpy(), faces)
                ogvs, ogfs, oges = mesh.export_soup()
                ogmesh = Mesh(ogvs, ogfs)

                vertices = torch.from_numpy(ogvs).to(device)
                faces = torch.from_numpy(ogfs).long().to(device)
                fverts = vertices[faces]

                if new_init:
                    rng = default_rng()
                    n_cuts = rng.integers(self.args.min_cuts, self.args.max_cuts+1)
                    cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)

                    # Unit test: mesh is still connected
                    vs, fs, es = mesh.export_soup()
                    testsoup = PolygonSoup(vs, fs)
                    n_components = testsoup.nConnectedComponents()
                    assert n_components == 1, f"After cutting found {n_components} components!"

                    # Save new topology
                    self.cutvs = vs
                    self.cutfs = fs

                    # Only replace SLIM if nan
                    newslim = torch.from_numpy(SLIM(mesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2
                    set_new_slim = False
                    if torch.all(~torch.isnan(newslim)):
                        self.slimuv = newslim

                        # Convert slim to 3-dim
                        self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        meshprocessor = MeshProcessor.MeshProcessor.meshprocessor_from_array(vs, fs, self.source_dir, self._SourceMesh__ttype, cpuonly=self.cpuonly, load_wks_samples=self._SourceMesh__use_wks, load_wks_centroids=self._SourceMesh__use_wks)
                        meshprocessor.prepare_temporary_differential_operators(self._SourceMesh__ttype)
                        poissonsolver = meshprocessor.diff_ops.poisson_solver

                        # NOTE: We sometimes have NaNs here!!
                        self.slimj = poissonsolver.jacobians_from_vertices(self.slimuv) # F x 3 x 3

                        if torch.any(~torch.isfinite(self.slimj)):
                            print("SLIM Jacobians have NaNs!")
                        else:
                            set_new_slim = True
                    # Otherwise, just use SLIM with no cutting
                    if not set_new_slim:
                        self.slimuv = torch.from_numpy(SLIM(ogmesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2

                        # Convert slim to 3-dim
                        self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        self.slimj = self.jacobians_from_vertices(self.slimuv) #  F x 3 x 3

                        # Reset cut topo to original topo
                        self.cutvs = vertices.detach().cpu().numpy()
                        self.cutfs = faces.detach().numpy()
                else:
                    self.slimuv = torch.from_numpy(SLIM(ogmesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2

                    # Convert slim to 3-dim
                    self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                    # Get Jacobians
                    self.slimj = self.jacobians_from_vertices(self.slimuv) #  F x 3 x 3

                    # Reset cut topo to original topo
                    self.cutvs = vertices.detach().cpu().numpy()
                    self.cutfs = faces.detach().numpy()

                # DEBUG: make sure we can get back the original UVs up to global translation
                # NOTE: We compare triangle centroids bc face indexing gets messed up after cutting
                fverts = torch.from_numpy(ogvs[ogfs])
                # pred_V = torch.einsum("abc,acd->abd", (self.slimj[0,:,:2,:], fverts)).transpose(1,2)
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.slimj[0,:,:2,:].transpose(2,1)))

                if new_init and self.init == "slim" and set_new_slim:
                    checkslim = self.slimuv[0,fs,:2]
                    self.slimfuv = self.slimuv[:,fs,:2] # B x F x 3 x 2
                else:
                    checkslim = self.slimuv[0,faces,:2]
                    self.slimfuv = self.slimuv[:,faces,:2] # B x F x 3 x 2

                # diff = pred_V - checkslim
                # diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                # torch.testing.assert_allclose(diff.float(), torch.zeros(diff.shape), rtol=1e-4, atol=1e-4)

                ## Save the global translations
                self.slimtranslate = (checkslim - pred_V)[:,:,:2]

                ## Compute ground truth if set
                if self.args.lossgt:
                    from numpy.linalg import pinv
                    invJ = pinv(self.slimj.detach().cpu().numpy()[:,:,:2,:]) # F x 3 x 2
                    self.gtJ = torch.from_numpy(gtJ @ invJ) # F x 2 x 2
                    checkJ = self.gtJ @ self.slimj[:,:,:2,:]
                    # np.testing.assert_allclose(checkJ, gtJ, atol=1e-5, rtol=1e-5)

                    torch.save(self.gtJ, os.path.join(self.source_dir, "tmpgtJ.pt"))

                # Cache everything
                torch.save(self.slimfuv, os.path.join(self.source_dir, "slimfuv.pt"))
                torch.save(self.slimuv, os.path.join(self.source_dir, "slimuv.pt"))
                torch.save(self.slimj, os.path.join(self.source_dir, "slimj.pt"))
                torch.save(self.slimtranslate, os.path.join(self.source_dir, "slimtranslate.pt"))

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['slimfuv'] = self.slimfuv
            self.__loaded_data['slimuv'] = self.slimuv
            self.__loaded_data['slimj'] = self.slimj
            self.__loaded_data['slimtranslate'] = self.slimtranslate
            if self.args.lossgt:
                self.__loaded_data['gtJ'] = self.gtJ

            if self.initjinput:
                self.centroids_and_normals = torch.cat([self.centroids_and_normals, self.slimj.reshape(len(self.centroids_and_normals), -1)], dim=1)

        elif self.init == "isometric":
            if os.path.exists(os.path.join(self.source_dir, "isofuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "isoj.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "isotranslate.pt")) and \
                not new_init:

                self.isofuv = torch.load(os.path.join(self.source_dir, "isofuv.pt"))
                self.isoj = torch.load(os.path.join(self.source_dir, "isoj.pt"))
                self.isotranslate = torch.load(os.path.join(self.source_dir, "isotranslate.pt"))
            else:
                from utils import get_local_tris

                vertices = self.source_vertices
                device = vertices.device
                faces = self.get_source_triangles()
                mesh = Mesh(vertices.detach().cpu().numpy(), faces)
                vs, fs, es = mesh.export_soup()
                vertices = torch.from_numpy(vs).to(device)
                faces = torch.from_numpy(fs).long().to(device)
                fverts = vertices[faces]

                # Random choice of local basis
                if new_init:
                    # Global rotation of initialization
                    if new_init == "global":
                        local_tris = get_local_tris(vertices, faces, basis=None) # F x 3 x 2
                        theta = np.random.uniform(low=0, high=2 * np.pi, size=1)
                        rotationmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        local_tris = (np.matmul(rotationmat.reshape(1, 2, 2), local_tris.transpose(2,1))).transpose(2,1) # F x 3 x 2

                    # Sample random basis per triangle
                    if new_init == "basis":
                        basistype = np.random.choice(6, size=len(faces))
                        local_tris = get_local_tris(vertices, faces, basis=basistype) # F x 3 x 2

                    # Randomly sample rotations
                    if new_init == "rot":
                        local_tris = get_local_tris(vertices, faces, basis=None) # F x 3 x 2
                        thetas = np.random.uniform(low=0, high=2 * np.pi, size=len(local_tris))
                        rotations = np.array([[[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] for theta in thetas])
                        local_tris = (np.matmul(rotations, local_tris.transpose(2,1))).transpose(2,1) # F x 3 x 2

                else:
                    local_tris = get_local_tris(vertices, faces) # F x 3 x 2

                # Randomly sample displacements
                # NOTE: might need to alter the scales here
                if self.args.noiseiso:
                    local_tris = local_tris + np.random.uniform(size=(local_tris.shape[0], 1, local_tris.shape[2])) # F x 3 x 2
                else:
                    self.isofuv = local_tris

                # Unit testing: face areas should be same as in 3D
                from meshing.mesh import Mesh
                from meshing.analysis import computeFaceAreas

                mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().numpy())
                computeFaceAreas(mesh)
                fareas3d = mesh.fareas
                fareas2d = 0.5 * np.abs(torch.linalg.det(torch.cat([torch.ones((len(self.isofuv), 1, 3)).float(), self.isofuv.transpose(2,1)], dim=1)).numpy())

                np.testing.assert_allclose(fareas3d, fareas2d, err_msg="Isometric embedding: all triangle areas should be same!")

                #### Get jacobians using gradient operator per triangle
                from igl import grad

                # Gradient operator (F*3 x V)
                # NOTE: need to compute this separately per triangle b/c of soup
                from scipy.sparse import bmat
                G = []
                for i in range(len(faces)):
                    G.append(grad(fverts[i].detach().cpu().numpy(), np.arange(3).reshape(1,3)))
                # Create massive sparse block diagonal matrix
                G = bmat([[None for _ in range(i)] + [G[i]] + [None for _ in range(i+1, len(G))] for i in range(len(G))])

                # Convert local tris to soup
                isosoup = self.isofuv.reshape(-1, 2).detach().cpu().numpy() # V x 2

                # Get Jacobians
                self.isoj = torch.from_numpy((G @ isosoup).reshape(local_tris.shape)).transpose(2,1) # F x 3 x 2

                ## Debugging: make sure we can get back the original UVs up to global translation
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.isoj.transpose(2,1)))
                diff = pred_V - self.isofuv
                diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle global translation
                torch.testing.assert_allclose(diff.float(), torch.zeros(diff.shape), rtol=1e-4, atol=1e-4)

                ## Save the global translations
                self.isotranslate = self.isofuv - pred_V

                ## Compute ground truth if set
                if self.args.lossgt:
                    from numpy.linalg import pinv
                    invJ = pinv(self.isoj.detach().cpu().numpy()) # F x 3 x 2
                    self.gtJ = torch.from_numpy(gtJ @ invJ) # F x 2 x 2
                    checkJ = self.gtJ @ self.isoj
                    # np.testing.assert_allclose(checkJ, gtJ, atol=1e-5, rtol=1e-5)

                    torch.save(self.gtJ, os.path.join(self.source_dir, "tmpgtJ.pt"))

                # Cache everything
                torch.save(self.isofuv, os.path.join(self.source_dir, "isofuv.pt"))
                torch.save(self.isoj, os.path.join(self.source_dir, "isoj.pt"))
                torch.save(self.isotranslate, os.path.join(self.source_dir, "isotranslate.pt"))

            ## Store in loaded data so it gets mapped to device
            # NOTE: need to transpose isoj to interpret as 2x3
            self.__loaded_data['isofuv'] = self.isofuv
            self.__loaded_data['isoj'] = self.isoj
            self.__loaded_data['isotranslate'] = self.isotranslate
            if self.args.lossgt:
                self.__loaded_data['gtJ'] = self.gtJ

            if self.initjinput:
                self.centroids_and_normals = torch.cat([self.centroids_and_normals, self.isoj.reshape(len(self.centroids_and_normals), -1)], dim=1)

        ### Dense: Use initialization jacobians as input
        if self.flatten == "input":
            if self.init == "tutte":
                self.flat_vector = self.tuttej.reshape(1, -1)
            elif self.init == "isometric":
                self.flat_vector = torch.cat([self.isoj, torch.zeros((self.isoj.shape[0], 1, 3))], dim=1).reshape(1, -1)
            elif self.init == "slim":
                self.flat_vector = self.slimj.reshape(1, -1)
            # nchannels = self.centroids_and_normals.shape[1]
            # gsize = int(np.ceil(nchannels/9))
            # newchannels = []
            # for i in range(9):
            #     newchannels.append(torch.sum(self.centroids_and_normals[:,i*gsize:(i+1)*gsize], dim=1))
            # self.flat_vector = torch.stack(newchannels, dim=1).reshape(1, -1)

        # Essentially here we load pointnet data and apply the same preprocessing
        for key in self.__extra_keys:
            data = self.mesh_processor.get_data(key)
            # if data is None:  # not found in mesh data so try loading from disk
            #     data = np.load(os.path.join(self.source_dir, key + ".npy"))
            data = torch.from_numpy(data)
            if key == 'samples':
                if self.center_source:
                    data -= self.get_mesh_centroid()
                scale = self.__random_scale
                data *= scale
            data = data.unsqueeze(0).type(self.__ttype)

            self.__loaded_data[key] = data
        # print("Ellapsed load source mesh ", time.time() - start)

    def load(self, source_v=None, source_f=None, new_init=False):
        if source_v is not None and source_f is not None:
            self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_array(source_v,source_f, self.source_dir, self.__ttype,
                                                                                       cpuonly=self.cpuonly, load_wks_samples=self.__use_wks,
                                                                                       load_wks_centroids=self.__use_wks,
                                                                                       top_k_eig=self.top_k_eig)
        else:
            if os.path.isdir(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_directory(self.source_dir, self.__ttype,
                                                                                               cpuonly=self.cpuonly,
                                                                                               load_wks_samples=self.__use_wks,
                                                                                               load_wks_centroids=self.__use_wks,
                                                                                               top_k_eig=self.top_k_eig)
            else:
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_file(self.source_dir, self.__ttype,
                                                                                          cpuonly=self.cpuonly,
                                                                                          load_wks_samples=self.__use_wks,
                                                                                          load_wks_centroids=self.__use_wks,
                                                                                          top_k_eig=self.top_k_eig)
        self.__init_from_mesh_data(new_init)

    def get_point_dim(self):
        if self.flatten:
            return self.flat_vector.shape[1]
        return self.centroids_and_normals.shape[1]

    def get_centroids_and_normals(self):
        return self.centroids_and_normals

    def get_mesh_centroid(self):
        return self.source_mesh_centroid

    def pin_memory(self):
        # self.poisson.pin_memory()
        # self.centroids_and_normals.pin_memory()
        # self.source_vertices.pin_memory()
        # for key in self.__loaded_data.keys():
        #     self.__loaded_data[key].pin_memory()
        return self