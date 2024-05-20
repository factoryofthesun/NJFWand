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
from diffusionnet import get_operators
from utils import FourierFeatureTransform, get_jacobian_torch
import dill as pickle

class SourceMesh:
    '''
    datastructure for the source mesh to be mapped
    '''

    def __init__(self, args, source_ind, source_dir, extra_source_fields,
                 random_scale, ttype, use_wks=False, random_centering=False,
                cpuonly=False, init=False, fft=None, fftscale=None, flatten=False,
                initjinput=False, debug=False, top_k_eig=50):
        self.args = args
        self.__use_wks = use_wks
        self.source_ind = source_ind
        # NOTE: This is the CACHE DIRECTORY
        self.source_dir = source_dir
        self.input_features = None
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
        self.initvs = None
        self.initfs = None
        self.initjinput = initjinput
        self.flatten = flatten
        self.debug = debug
        self.top_k_eig = top_k_eig
        self.fgroups = None

        self.fft = None
        if fft:
            # Compute input channels
            if self.__use_wks:
                n_input = 106
            else:
                n_input = 6

            # Initialize fourier features transform
            self.fft = FourierFeatureTransform(n_input, fft, fftscale)

        self.initweights = None

        ### MeshCNN stuff
        if self.args.arch in ["fullconv", 'directedge', 'fullconv2']:
            self.pool_count = 0
            self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
            self.edges = self.gemm_edges = self.sides = None
            self.history_data = None

    def get_vertices(self):
        return self.source_vertices

    def get_global_translation_to_original(self):
        return self.__source_global_translation_to_original

    def vertices_from_jacobians(self, d, updatedlap=False):
        return self.poisson.solve_poisson(d, updatedlap=updatedlap)

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
        self.input_features = self.input_features.to(device)
        for key in self.__loaded_data.keys():
            self.__loaded_data[key] = self.__loaded_data[key].to(device)
        return self

    ### PRECOMPUTATION HAPPENS HERE ###
    def __init_from_mesh_data(self, new_init=False):
        from meshing.mesh import Mesh
        from meshing.edit import EdgeCut
        from meshing.io import PolygonSoup
        import matplotlib.pyplot as plt

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

        vertices = self.source_vertices
        device = vertices.device
        faces = torch.from_numpy(self.get_source_triangles()).long()
        self.faces = faces

        #### PROCESS INPUT FEATURES ####
        if self.args.arch in ["diffusionnet", "fullconv", "directedge", "directuv", "fullconv2"]:
            """ (points, wks) over vertices """
            centroids = self.mesh_processor.get_centroids()
            c = self.source_mesh_centroid
            if self.center_source:
                self.source_vertices -= c
                self.__source_global_translation_to_original = c

            # By default always use wks features
            if self.args.inittopo and self.args.init == "precut":

                # if os.path.exists(os.path.join(self.source_dir, "input_features_inittopo.pt")):
                #     os.remove(os.path.join(self.source_dir, "input_features_inittopo.pt"))

                # Check for cached features
                if os.path.exists(os.path.join(self.source_dir, "input_features_inittopo.pt")) and \
                    os.path.exists(os.path.join(self.source_dir, "initvs.pt")) and \
                    os.path.exists(os.path.join(self.source_dir, "initfs.pt")) and \
                    os.path.exists(os.path.join(self.source_dir, "fuv_to_v.npy")):

                    self.input_features = torch.load(os.path.join(self.source_dir, "input_features_inittopo.pt"))
                    self.initvs = torch.load(os.path.join(self.source_dir, "initvs.pt"))
                    self.initfs = torch.load(os.path.join(self.source_dir, "initfs.pt"))
                    self.fuv_to_v = np.load(os.path.join(self.source_dir, "fuv_to_v.npy"))

                    self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = get_operators(self.initvs, self.initfs,
                                                                                op_cache_dir=self.source_dir, k_eig=self.top_k_eig)
                    self.frames.to(device)
                    self.mass.to(device)
                    self.L.to(device)
                    self.evals.to(device)
                    self.evecs.to(device)
                    self.gradX.to(device)
                    self.gradY.to(device)

                    self.__loaded_data['faces'] = self.faces
                    self.__loaded_data['frames'] = self.frames
                    self.__loaded_data['mass'] = self.mass
                    self.__loaded_data['L'] = self.L
                    self.__loaded_data['evals'] = self.evals
                    self.__loaded_data['evecs'] = self.evecs
                    self.__loaded_data['gradX'] = self.gradX
                    self.__loaded_data['gradY'] = self.gradY
                else:
                    # Use init topology to compute features
                    cutsoup = PolygonSoup.from_obj(os.path.join(self.source_dir, "..", "..", f"{self.source_ind}.obj"))
                    initfs = cutsoup.face_uv.astype(int)

                    fuv_to_v = np.zeros(np.max(initfs) + 1).astype(int)
                    fuv_to_v[initfs.flatten()] = cutsoup.indices.flatten()

                    initfs_ogidx = fuv_to_v[np.sort(np.unique(initfs))]
                    initvs = vertices[initfs_ogidx].detach().cpu().numpy()
                    self.mesh_processor.computeWKS(vs = initvs, fs = initfs)

                    wks = WKS_FACTOR * self.mesh_processor.vert_wks
                    self.input_features = torch.from_numpy(wks).type(self.__ttype)

                    # Get geometric operators
                    initvs = torch.from_numpy(initvs)
                    initfs = torch.from_numpy(initfs)

                    self.initvs = initvs
                    self.initfs = initfs
                    self.fuv_to_v = fuv_to_v

                    self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = get_operators(initvs, initfs,
                                                                                op_cache_dir=self.source_dir, k_eig=self.top_k_eig)
                    self.frames.to(device)
                    self.mass.to(device)
                    self.L.to(device)
                    self.evals.to(device)
                    self.evecs.to(device)
                    self.gradX.to(device)
                    self.gradY.to(device)

                    self.__loaded_data['faces'] = self.faces
                    self.__loaded_data['frames'] = self.frames
                    self.__loaded_data['mass'] = self.mass
                    self.__loaded_data['L'] = self.L
                    self.__loaded_data['evals'] = self.evals
                    self.__loaded_data['evecs'] = self.evecs
                    self.__loaded_data['gradX'] = self.gradX
                    self.__loaded_data['gradY'] = self.gradY

                    # Heat kernel signature
                    if self.args.hks:
                        from diffusionnet.geometry import compute_hks_autoscale
                        hks = compute_hks_autoscale(self.evals, self.evecs, 16)
                        self.input_features = torch.cat((self.input_features, hks.type(self.__ttype)), dim=1)

                    # Cache
                    if not self.args.debug:
                        torch.save(self.input_features, os.path.join(self.source_dir, "input_features_inittopo.pt"))
                        torch.save(initvs, os.path.join(self.source_dir, "initvs.pt"))
                        torch.save(initfs, os.path.join(self.source_dir, "initfs.pt"))
                        np.save(os.path.join(self.source_dir, "fuv_to_v.npy"), self.fuv_to_v)

            else:
                featurename = "input_features"
                if self.args.extrinsic:
                    if self.args.fftextrinsic:
                        featurename += "_fftextrinsic"
                    else:
                        featurename += "_extrinsic"
                if os.path.exists(os.path.join(self.source_dir, f"{featurename}.pt")) and \
                    (os.path.exists(os.path.join(self.source_dir, "vertexnormals.pt")) or not self.args.extrinsic) and \
                        not self.args.debug:
                    self.input_features = torch.load(os.path.join(self.source_dir, f"{featurename}.pt"))

                    # Get geometric operators
                    self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = get_operators(self.source_vertices, faces,
                                                                                op_cache_dir=self.source_dir, k_eig=self.top_k_eig)
                    self.frames.to(device)
                    self.mass.to(device)
                    self.L.to(device)
                    self.evals.to(device)
                    self.evecs.to(device)
                    self.gradX.to(device)
                    self.gradY.to(device)

                    self.__loaded_data['faces'] = self.faces
                    self.__loaded_data['frames'] = self.frames
                    self.__loaded_data['mass'] = self.mass
                    self.__loaded_data['L'] = self.L
                    self.__loaded_data['evals'] = self.evals
                    self.__loaded_data['evecs'] = self.evecs
                    self.__loaded_data['gradX'] = self.gradX
                    self.__loaded_data['gradY'] = self.gradY

                    if self.args.fftextrinsic and self.args.extrinsic:
                        self.vertexnormals = torch.load(os.path.join(self.source_dir, "vertexnormals.pt"))
                        self.__loaded_data['vertexnormals'] = self.vertexnormals
                else:
                    self.mesh_processor.computeWKS()

                    wks = WKS_FACTOR * self.mesh_processor.vert_wks
                    self.input_features = torch.from_numpy(wks).type(self.__ttype)

                    # Get geometric operators
                    self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = get_operators(self.source_vertices, faces,
                                                                                op_cache_dir=self.source_dir, k_eig=self.top_k_eig)
                    self.frames.to(device)
                    self.mass.to(device)
                    self.L.to(device)
                    self.evals.to(device)
                    self.evecs.to(device)
                    self.gradX.to(device)
                    self.gradY.to(device)

                    self.__loaded_data['faces'] = self.faces
                    self.__loaded_data['frames'] = self.frames
                    self.__loaded_data['mass'] = self.mass
                    self.__loaded_data['L'] = self.L
                    self.__loaded_data['evals'] = self.evals
                    self.__loaded_data['evecs'] = self.evecs
                    self.__loaded_data['gradX'] = self.gradX
                    self.__loaded_data['gradY'] = self.gradY

                    # Heat kernel signature
                    if self.args.hks:
                        from diffusionnet.geometry import compute_hks_autoscale
                        hks = compute_hks_autoscale(self.evals, self.evecs, 16)
                        self.input_features = torch.cat((self.input_features, hks.type(self.__ttype)), dim=1)

                    # Extrinsic features: xyz, normals
                    # NOTE: If fft on extrinsics, then we take care of this in the network
                    if self.args.extrinsic and not self.args.fftextrinsic:
                        self.input_features = torch.cat((self.input_features, self.source_vertices.clone().type(self.__ttype)), dim=1)
                        from meshing.analysis import computeVertexNormals
                        mesh = Mesh(self.source_vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
                        computeVertexNormals(mesh)
                        self.vertexnormals = torch.from_numpy(mesh.vertexnormals).type(self.__ttype)
                        self.input_features = torch.cat((self.input_features, torch.from_numpy(mesh.vertexnormals).type(self.__ttype)), dim=1)
                    elif self.args.extrinsic:
                        from meshing.analysis import computeVertexNormals
                        mesh = Mesh(self.source_vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
                        computeVertexNormals(mesh)
                        self.vertexnormals = torch.from_numpy(mesh.vertexnormals).type(self.__ttype)
                        self.__loaded_data['vertexnormals'] = self.vertexnormals

                    if not self.args.debug:
                        torch.save(self.input_features, os.path.join(self.source_dir, f"{featurename}.pt"))

                        # Also save vertexnormals if necessary
                        if self.args.extrinsic:
                            torch.save(self.vertexnormals, os.path.join(self.source_dir, "vertexnormals.pt"))

        elif self.args.arch == "mlp":
            """ (centroids, normals, wks) over faces """
            centroids = self.mesh_processor.get_centroids()
            centroid_points_and_normals = centroids.points_and_normals
            if self.__use_wks:
                wks = WKS_FACTOR * centroids.wks
                centroid_points_and_normals = numpy.hstack((centroid_points_and_normals, wks))
            self.input_features = torch.from_numpy(
                centroid_points_and_normals).type(self.__ttype)
            if self.center_source:
                c = self.source_mesh_centroid
                self.input_features[:, 0:3] -= c
                self.source_vertices -= c
                self.__source_global_translation_to_original = c

            # Get geometric operators
            self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = get_operators(self.source_vertices, faces,
                                                                        op_cache_dir=self.source_dir, k_eig=self.top_k_eig)
            self.frames.to(device)
            self.mass.to(device)
            self.L.to(device)
            self.evals.to(device)
            self.evecs.to(device)
            self.gradX.to(device)
            self.gradY.to(device)

            self.__loaded_data['faces'] = self.faces
            self.__loaded_data['frames'] = self.frames
            self.__loaded_data['mass'] = self.mass
            self.__loaded_data['L'] = self.L
            self.__loaded_data['evals'] = self.evals
            self.__loaded_data['evecs'] = self.evecs
            self.__loaded_data['gradX'] = self.gradX
            self.__loaded_data['gradY'] = self.gradY

            # Heat kernel signature
            if self.args.hks:
                from diffusionnet.geometry import compute_hks_autoscale
                hks = compute_hks_autoscale(self.evals, self.evecs, 16)

                # Average over faces
                fhks = torch.mean(hks[faces], dim=1)

                self.input_features = torch.cat((self.input_features, fhks.type(self.__ttype)), dim=1)

            if self.fft is not None:
                self.input_features = self.fft(self.input_features)

        else:
            raise ValueError(f"Architecture {self.args.arch} not implemented!")


        self.poisson = self.mesh_processor.diff_ops.poisson_solver

        ## Precompute edge lengths and edgeidxs
        from source_njf.utils import get_edge_pairs, vertex_soup_correspondences, edge_soup_correspondences
        from itertools import combinations

        # Reset cut topo to original topo
        self.cutvs = vertices.detach().cpu().numpy()
        self.cutfs = faces.detach().cpu().numpy()
        ogvs = vertices.detach().cpu().numpy()
        ogfs = faces.detach().cpu().numpy()

        if os.path.exists(os.path.join(self.source_dir, "edge_vpairs.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "facepairs_nobound.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "elens_nobound.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "ogedge_vpairs_nobound.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "valid_pairs.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "valid_edge_pairs.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "facepairs.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "allfacepairs.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "valid_elens.pt")) and \
            os.path.exists(os.path.join(self.source_dir, "edgecorrespondences.pkl")) and \
            os.path.exists(os.path.join(self.source_dir, "meshe_to_vpair.pkl")) and \
            os.path.exists(os.path.join(self.source_dir, "vpair_to_meshe.pkl")) and \
            os.path.exists(os.path.join(self.source_dir, "meshe_to_meshenobound.pkl")):

            self.edge_vpairs = torch.load(os.path.join(self.source_dir, "edge_vpairs.pt"))
            self.facepairs_nobound = torch.load(os.path.join(self.source_dir, "facepairs_nobound.pt"))
            self.elens_nobound = torch.load(os.path.join(self.source_dir, "elens_nobound.pt"))
            self.ogedge_vpairs_nobound = torch.load(os.path.join(self.source_dir, "ogedge_vpairs_nobound.pt"))
            self.valid_pairs = torch.load(os.path.join(self.source_dir, "valid_pairs.pt"))
            self.valid_edge_pairs = torch.load(os.path.join(self.source_dir, "valid_edge_pairs.pt"))
            self.facepairs = torch.load(os.path.join(self.source_dir, "facepairs.pt"))
            self.allfacepairs = torch.load(os.path.join(self.source_dir, "allfacepairs.pt"))
            self.valid_elens = torch.load(os.path.join(self.source_dir, "valid_elens.pt"))
            # self.edges = torch.load(os.path.join(self.source_dir, "edges.pt"))

            import dill as pickle
            self.edgecorrespondences = pickle.load(open(os.path.join(self.source_dir, "edgecorrespondences.pkl"), "rb"))
            self.meshe_to_vpair = pickle.load(open(os.path.join(self.source_dir, "meshe_to_vpair.pkl"), "rb"))
            self.vpair_to_meshe = pickle.load(open(os.path.join(self.source_dir, "vpair_to_meshe.pkl"), "rb"))
            self.meshe_to_meshenobound = pickle.load(open(os.path.join(self.source_dir, "meshe_to_meshenobound.pkl"), "rb"))
        else:
            print(f"Constructing all correspondence/topology variables ...")
            vcorrespondences = vertex_soup_correspondences(faces)
            valid_pairs = []
            for ogv, vlist in sorted(vcorrespondences.items()):
                valid_pairs.extend(list(combinations(vlist, 2)))
            self.valid_pairs = valid_pairs

            # Get face pairs for all corresponding valid pairs
            self.allfacepairs = []
            for pair in valid_pairs:
                self.allfacepairs.append([pair[0] // 3, pair[1] // 3])

            # There should be no self-pairs
            for pair in self.allfacepairs:
                assert pair[0] != pair[1], f"Self-pair found: {pair}"

            mesh = Mesh(ogvs, ogfs)
            self.valid_edge_pairs, self.valid_edges_to_soup, self.edgeidxs, self.edgededupidxs, edges, self.valid_elens, self.facepairs = get_edge_pairs(mesh, valid_pairs, device=device)

            # Convert edge pairs to tensor
            self.valid_pairs = torch.tensor([list(pair) for pair in self.valid_pairs], device=device)
            self.valid_edge_pairs = torch.tensor([list(pair) for pair in self.valid_edge_pairs], device=device)
            self.facepairs = torch.tensor(self.facepairs, device=device)
            self.allfacepairs = torch.tensor(self.allfacepairs, device=device)
            self.valid_elens = self.valid_elens.to(device)
            # self.edges = np.array(self.edges)

            ### Edge correspondences
            from source_njf.utils import meshe_to_vpair
            self.edgecorrespondences, self.facecorrespondences = edge_soup_correspondences(ogfs)
            self.meshe_to_vpair = meshe_to_vpair(mesh)
            self.vpair_to_meshe = {v: k for k, v in self.meshe_to_vpair.items()} # Reverse for vpair to meshe lookup

            # Get corresponding edge lengths, og edge pairs, and face correspondences w/o boundary
            elens_nobound = []
            edge_vpairs = []
            # NOTE: This maps eidx => new edge indexing with boundaries removed == self.initweights index!!!!
            self.meshe_to_meshenobound = {}

            self.ogedge_vpairs_nobound = []
            self.facepairs_nobound = []
            count = 0
            for eidx, ogvpair in sorted(self.meshe_to_vpair.items()):
                soupvpairs = self.edgecorrespondences[ogvpair]
                fpair = self.facecorrespondences[ogvpair]
                ogvpair = list(ogvpair)

                if len(soupvpairs) == 1:
                    continue

                elens_nobound.append(np.linalg.norm(ogvs[ogvpair[1]] - ogvs[ogvpair[0]]))

                # Sanity check edge length
                np.testing.assert_almost_equal(elens_nobound[-1], mesh.length(mesh.topology.edges[eidx]))

                edge_vpairs.append(soupvpairs)
                self.meshe_to_meshenobound[eidx] = count
                self.ogedge_vpairs_nobound.append(ogvpair)

                # Get face pair
                assert len(fpair) == 2, f"Edge corresponding face pair {fpair} does not have 2 faces!"
                self.facepairs_nobound.append(fpair)

                count += 1

            # NOTE: All are sorted by edges NOT on boundary!!
            self.ogedge_vpairs_nobound = torch.tensor(self.ogedge_vpairs_nobound, device=device).long() # E x 2
            self.elens_nobound = torch.tensor(elens_nobound, device=device)
            self.facepairs_nobound = torch.tensor(self.facepairs_nobound, device=device).long()

            # NOTE: FIRST square dimension gives the corresponding vertices across the two soup edges
            edge_vpairs = np.array(edge_vpairs).transpose(0,2,1) # E x 2 x 2 (edges x (edge 1 v1, edge 1 v2) x (edge 2 v1, edge 2 v2)
            self.edge_vpairs = torch.from_numpy(edge_vpairs).to(device).long()

            assert len(self.edge_vpairs) == len(self.elens_nobound) == len(self.ogedge_vpairs_nobound) == len(self.facepairs_nobound), f"Edge pairs {len(self.edge_vpairs)}, edge lengths {len(self.elens_nobound)}, og edge pairs {len(self.ogedge_vpairs_nobound)}, and face pairs {len(self.facepairs_nobound)} do not have the same length!"

            torch.save(self.edge_vpairs, os.path.join(self.source_dir, "edge_vpairs.pt"))
            torch.save(self.facepairs_nobound, os.path.join(self.source_dir, "facepairs_nobound.pt"))
            torch.save(self.elens_nobound, os.path.join(self.source_dir, "elens_nobound.pt"))
            torch.save(self.ogedge_vpairs_nobound, os.path.join(self.source_dir, "ogedge_vpairs_nobound.pt"))
            torch.save(self.valid_pairs, os.path.join(self.source_dir, "valid_pairs.pt"))
            torch.save(self.valid_edge_pairs, os.path.join(self.source_dir, "valid_edge_pairs.pt"))
            torch.save(self.facepairs, os.path.join(self.source_dir, "facepairs.pt"))
            torch.save(self.allfacepairs, os.path.join(self.source_dir, "allfacepairs.pt"))
            torch.save(self.valid_elens, os.path.join(self.source_dir, "valid_elens.pt"))

            import dill as pickle
            pickle.dump(self.edgecorrespondences, open(os.path.join(self.source_dir, "edgecorrespondences.pkl"), "wb"))
            pickle.dump(self.meshe_to_vpair, open(os.path.join(self.source_dir, "meshe_to_vpair.pkl"), "wb"))
            pickle.dump(self.vpair_to_meshe, open(os.path.join(self.source_dir, "vpair_to_meshe.pkl"), "wb"))
            pickle.dump(self.meshe_to_meshenobound, open(os.path.join(self.source_dir, "meshe_to_meshenobound.pkl"), "wb"))
            print(f"Variables saved in {self.source_dir}")

        if self.args.arch in ["fullconv", 'directedge', 'fullconv2']:
            load_path = os.path.join(self.source_dir, "meshcnn_features.npz")

            if os.path.exists(load_path) and os.path.exists(os.path.join(self.source_dir, "ve.pkl")) and \
                os.path.exists(os.path.join(self.source_dir, "edge2key.pkl")):
                mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)

                with open(os.path.join(self.source_dir, "ve.pkl"), "rb") as f:
                    mesh_data.ve = pickle.load(f)

                with open(os.path.join(self.source_dir, "edge2key.pkl"), "rb") as f:
                    mesh_data.edge2key = pickle.load(f)
            else:
                class MeshPrep:
                    def __getitem__(self, item):
                        return eval('self.' + item)

                mesh_data = MeshPrep()
                mesh_data.vs = mesh_data.edges = None
                mesh_data.gemm_edges = mesh_data.sides = None
                mesh_data.edges_count = None
                mesh_data.ve = None
                mesh_data.v_mask = None
                mesh_data.filename = 'unknown'
                mesh_data.edge_lengths = None
                mesh_data.edge_areas = []
                mesh_data.vs = vertices.detach().cpu().numpy()
                mesh_data.faces = faces.detach().cpu().numpy()
                mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)

                # faces, face_areas = remove_non_manifolds(mesh_data, faces)
                from source_njf.utils import build_gemm
                from meshing.analysis import computeFaceAreas
                mesh = Mesh(mesh_data.vs, mesh_data.faces)
                computeFaceAreas(mesh)
                build_gemm(mesh_data, mesh_data.faces, mesh.fareas)
                self.edge2key = mesh_data.edge2key

                ### UNIT TEST: we use mesh_data.edges to map from meshcnn eidx to our mesh eidx
                # TODO: check gemm_edges

                # VE should map vertices to the same edges b/w meshcnn and mesh (assert based on vertex indices)
                ve = mesh_data.ve
                edges = mesh_data.edges # tuple(v1, v2) => meshcnn eidx order
                for vi, ei_list in enumerate(ve):
                    og_e_list = [e for e in mesh.topology.vertices[vi].adjacentEdges()]
                    og_e_vertices_list = [tuple(sorted([e.halfedge.vertex.index, e.halfedge.twin.vertex.index])) for e in og_e_list]
                    compare_e_vertices_list = [tuple(edges[eidx]) for eidx in ei_list]

                    assert len(og_e_vertices_list) == len(compare_e_vertices_list), f"Vertex {vi} has different number of incident edges in meshcnn and mesh!"

                    for og_e_vertices in og_e_vertices_list:
                        assert og_e_vertices in compare_e_vertices_list, f"Vertex {vi}, og edge {og_e_vertices} not found in meshcnn edges list."

                ### gfmm (3 face neighbors)
                self.gfmm = []
                for fidx, face in sorted(mesh.topology.faces.items()):
                    neighbors = [f.index + 1 for f in face.adjacentFaces()] # NOTE: +1 due to 0-padding in first row

                    # NOTE: we pad using the 0 index (first feature index will be all 0s)
                    if len(neighbors) < 3:
                        neighbors += [0] * (3 - len(neighbors))

                    self.gfmm.append(neighbors)
                self.gfmm = np.array(self.gfmm, dtype=int)
                mesh_data.gfmm = self.gfmm

                np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                                    edges_count=mesh_data.edges_count, v_mask=mesh_data.v_mask,
                                    filename=mesh_data.filename, sides=mesh_data.sides,
                                    edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                                    gfmm = self.gfmm)

                # NOTE: ve is a jagged mapping (maps vertices to all incident edge indices)
                import dill as pickle

                with open(os.path.join(self.source_dir, "ve.pkl"), "wb") as f:
                    pickle.dump(mesh_data.ve, f)

                with open(os.path.join(self.source_dir, "edge2key.pkl"), "wb") as f:
                    pickle.dump(self.edge2key, f)

            self.vs = mesh_data['vs']
            self.edges = mesh_data['edges']
            self.gemm_edges = mesh_data['gemm_edges']
            self.edges_count = int(mesh_data['edges_count'])
            self.ve = mesh_data.ve
            self.v_mask = mesh_data['v_mask']
            self.filename = str(mesh_data['filename'])
            self.edge_lengths = mesh_data['edge_lengths']
            self.edge_areas = mesh_data['edge_areas']
            self.sides = mesh_data['sides']
            self.gfmm = mesh_data['gfmm']
            self.edge2key = mesh_data.edge2key

            self.init_history()

            ### Precompute MeshCNN features ###
            # self.edges_count = len(elens_nobound) # NOTE: only count edges not on boundary
            # self.gemm_edges = [] # E x 4 for four edge neighbors
            # eidxs_nobound = np.sort(np.unique(list(self.meshe_to_meshenobound.values())))

            # for eidx in eidxs_nobound:
            #     # Get edge neighbors -- excluding itself!!
            #     edges = mesh.topology.edges[eidx]
            #     neighbors = [edges.halfedge.next.edge.index, edges.halfedge.next.next.edge.index,
            #                     edges.halfedge.twin.next.edge.index, edges.halfedge.twin.next.next.edge.index]
            #     assert len(np.unique(neighbors)) == len(neighbors), f"Edge {eidx} has duplicate neighbors!"
            #     self.gemm_edges.append(neighbors)
            # self.gemm_edges = torch.tensor(self.gemm_edges, device=device).long()

            # # Set sides indexing (where each eidx is in relative the four neighbor gemm idxs)
            # self.sides = []
            # for i, neighbors in enumerate(self.gemm_edges):
            #     self.sides.append(torch.where(self.gemm_edges[neighbors] == i)[1])
            # self.sides = torch.concatenate(self.sides, dim=0).long()

        ### NOTE: BASE WEIGHTS INITIALIZED HERE
        if self.args.spweight == "sigmoid":
            if self.init == "isometric":
                self.initweights = torch.ones(len(self.edge_vpairs), device=device).double() * -10
            else:
                self.initweights = torch.ones(len(self.edge_vpairs), device=device).double()
        elif self.args.spweight in ["softmax", "nonzero"]:
            self.initweights = torch.ones(len(self.edge_vpairs), device=device).double() * 100
        elif self.args.spweight in ["seamless", 'cosine']:
            if self.init == "isometric":
                self.initweights = torch.zeros(len(self.edge_vpairs), device=device).double()
            else:
                self.initweights = torch.ones(len(self.edge_vpairs), device=device).double() * 0.5
        else:
            raise NotImplementedError(f"Soft poisson weight {self.args.spweight} not implemented!")

        # Also define cut flags
        self.initcuts = torch.zeros(len(self.edge_vpairs), device=device)

        ### Render the 3D mesh colored by 3D position (to match against the UVs)
        if not os.path.exists(os.path.join(self.source_dir, "..", "..", f"{self.source_ind}_3dpos.png")):
            from results_saving_scripts.plot_uv import export_views

            vpos = vertices[faces].detach().cpu().numpy().reshape(-1, 3)
            vpos -= np.min(vpos, axis=0, keepdims=True)
            vpos /= np.max(np.linalg.norm(vpos, axis=1))

            export_views(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(),
                         os.path.join(self.source_dir, "..", ".."),
                        filename=f"{self.source_ind}_3dpos.png",
                            plotname=f"3D Mesh (Colored by Vertex Position)",
                            vcolors=vpos,
                            outline_width=0.01, cmap = plt.get_cmap('Reds'),
                            device="cpu", n_sample=30, width=200, height=200,
                            vmin=0, vmax=1, shading=False)

        ### Load ground truth UVs if set
        if self.args.gtuvloss:
            self.gt_uvs = torch.load(os.path.join(self.source_dir, "..", "..", "gt_uvs.pt"))
            # Center the UVs
            self.gt_uvs -= torch.mean(self.gt_uvs, dim=0, keepdim=True)
            self.__loaded_data['gt_uvs'] = self.gt_uvs

            # Load fgroups (if available)
            fgroupdir = os.path.join(self.source_dir, "..", "..", "fgroups.npy")
            if os.path.exists(fgroupdir):
                self.fgroups = np.load(fgroupdir)

            # Plot GT UVs (for visualization)
            from matplotlib.tri import Triangulation
            import matplotlib.pyplot as plt

            if self.fgroups is not None:
                for fgroup in np.unique(self.fgroups):
                    if os.path.exists(os.path.join(self.source_dir, "..", "..", f"gtuv_{fgroup}.png")):
                        continue

                    fgroupmask = self.fgroups == fgroup

                    groupuvs = self.gt_uvs.reshape(-1, 3, 2)[fgroupmask].reshape(-1, 2)
                    triangles = np.arange(len(groupuvs)).reshape(-1, 3)
                    tris = Triangulation(groupuvs[:, 0], groupuvs[:, 1], triangles=triangles)
                    fig, axs = plt.subplots(figsize=(5, 5))
                    axs.set_title(f"{self.source_ind} Ground Truth UVs (fgroup {fgroup})")

                    # # Color based on 3D positions
                    # fvs = mesh.vertices[mesh.faces] # F x 3 x 3
                    # fpos = np.mean(fvs, axis=1)
                    # fpos -= np.min(fpos, axis=0, keepdims=True)
                    # fpos /= np.max(np.linalg.norm(fpos, axis=1))

                    # from matplotlib.collections import PolyCollection

                    # colors = facecolors
                    # maskedTris = tris.get_masked_triangles()
                    # verts = np.stack((tris.x[maskedTris], tris.y[maskedTris]), axis=-1)
                    # collection = PolyCollection(verts)
                    # collection.set_facecolor(colors)
                    # axs.add_collection(collection)
                    # axs.autoscale_view()

                    cmap = plt.get_cmap("tab20")
                    axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                        linewidth=0.1, edgecolor="black")

                    plt.axis('off')
                    axs.axis('equal')
                    plt.savefig(os.path.join(self.source_dir, "..", "..", f"gtuv_{fgroup}.png"))
                    plt.close(fig)
                    plt.cla()
            else:
                if not os.path.exists(os.path.join(self.source_dir, "..", "..", f"gtuv.png")):
                    triangles = np.arange(len(self.gt_uvs)).reshape(-1, 3)
                    tris = Triangulation(self.gt_uvs[:, 0], self.gt_uvs[:, 1], triangles=triangles)
                    fig, axs = plt.subplots(figsize=(5, 5))
                    axs.set_title("Ground Truth UVs")
                    cmap = plt.get_cmap("tab20")
                    axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                        linewidth=0.1, edgecolor="black")
                    plt.axis('off')
                    axs.axis('equal')
                    plt.savefig(os.path.join(self.source_dir, "..", "..", "gtuv.png"))
                    plt.close(fig)
                    plt.cla()

            # Also plot GT cuts
            if os.path.exists(os.path.join(self.source_dir, "..", "..", "gt_cutvpairs.pkl")):
                import dill as pickle

                with open(os.path.join(self.source_dir, "..", "..", "gt_cutvpairs.pkl"), 'rb') as f:
                    self.gt_cutvpairs = pickle.load(f)

                if not os.path.exists(os.path.join(self.source_dir, "..", "..", "gtcuts.png")):
                    from results_saving_scripts.plot_uv import export_views
                    cutvs = [list(vpair) for vpair in self.gt_cutvpairs]
                    cylinderpos = ogvs[cutvs]
                    cylindervals = np.ones((len(cylinderpos), 2)) # E x 2

                    if len(cutvs) == 0:
                        cylinderpos = None
                        cylindervals = None

                    export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                                filename=f"gtcuts.png",
                                    plotname=f"Ground Truth Cuts", cylinders=cylinderpos,
                                    cylinder_scalars=cylindervals,
                                    outline_width=0.01, cmap = plt.get_cmap('Reds'),
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)

        ### Load ground truth jacobians + weights if set
        if self.args.gtnetworkloss or self.args.gtjacobianloss:
            self.gt_jacobians = torch.load(os.path.join(self.source_dir, "..", "..", "gt_j.pt"))

            # GT weights (need to be mapped to network output)
            with open(os.path.join(self.source_dir, "..", "..", "gt_cutvpairs.pkl"), 'rb') as f:
                self.gt_cutvpairs = pickle.load(f)

            # NOTE: network output is ordered by edge indexing
            if self.args.spweight == "sigmoid":
                self.gt_weights = torch.ones(len(self.edge_vpairs), device=device).double()
            elif self.args.spweight == "nonzero":
                self.gt_weights = torch.ones(len(self.edge_vpairs), device=device).double() * 1000

            # NOTE: sometimes we accidentally saved cut vpair indices that are on boundary of original mesh -- oops
            cut_es = [self.meshe_to_meshenobound[self.vpair_to_meshe[frozenset(vpair)]] for vpair in self.gt_cutvpairs if self.vpair_to_meshe[frozenset(vpair)] in self.meshe_to_meshenobound.keys()]
            self.gt_weights[cut_es] = 1e-7
            self.__loaded_data['gt_weights'] = self.gt_weights
            self.__loaded_data['gt_jacobians'] = self.gt_jacobians

            # Load fgroups (if available)
            fgroupdir = os.path.join(self.source_dir, "..", "..", "fgroups.npy")
            if os.path.exists(fgroupdir):
                self.fgroups = np.load(fgroupdir)

            # Plot GT UVs
            from matplotlib.tri import Triangulation
            import matplotlib.pyplot as plt
            if os.path.exists(os.path.join(self.source_dir, "..", "..", "gt_uvs.pt")):
                # Also load the gt uvs (for visualization)
                self.gt_uvs = torch.load(os.path.join(self.source_dir, "..", "..", "gt_uvs.pt"))
                # Center the UVs
                self.gt_uvs -= torch.mean(self.gt_uvs, dim=0, keepdim=True)

                # Load fgroups (if available)
                fgroupdir = os.path.join(self.source_dir, "..", "..", "fgroups.npy")
                if os.path.exists(fgroupdir):
                    self.fgroups = np.load(fgroupdir)

                # Plot GT UVs (for visualization)
                from matplotlib.tri import Triangulation
                import matplotlib.pyplot as plt

                if self.fgroups is not None:
                    for fgroup in np.unique(self.fgroups):
                        if os.path.exists(os.path.join(self.source_dir, "..", "..", f"gtuv_{fgroup}.png")):
                            continue
                        fgroupmask = self.fgroups == fgroup

                        groupuvs = self.gt_uvs.reshape(-1, 3, 2)[fgroupmask].reshape(-1, 2)
                        triangles = np.arange(len(groupuvs)).reshape(-1, 3)
                        tris = Triangulation(groupuvs[:, 0], groupuvs[:, 1], triangles=triangles)
                        fig, axs = plt.subplots(figsize=(5, 5))
                        axs.set_title(f"{self.source_ind} Ground Truth UVs (fgroup {fgroup})")

                        cmap = plt.get_cmap("tab20")
                        axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                            linewidth=0.1, edgecolor="black")

                        plt.axis('off')
                        axs.axis('equal')
                        plt.savefig(os.path.join(self.source_dir, "..", "..", f"gtuv_{fgroup}.png"))
                        plt.close(fig)
                        plt.cla()
                else:
                    if not os.path.exists(os.path.join(self.source_dir, "..", "..", f"gtuv.png")):
                        triangles = np.arange(len(self.gt_uvs)).reshape(-1, 3)
                        tris = Triangulation(self.gt_uvs[:, 0], self.gt_uvs[:, 1], triangles=triangles)
                        fig, axs = plt.subplots(figsize=(5, 5))
                        axs.set_title("Ground Truth UVs")
                        cmap = plt.get_cmap("tab20")
                        axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                            linewidth=0.1, edgecolor="black")
                        plt.axis('off')
                        axs.axis('equal')
                        plt.savefig(os.path.join(self.source_dir, "..", "..", "gtuv.png"))
                        plt.close(fig)
                        plt.cla()

            # Also plot GT cuts
            from results_saving_scripts.plot_uv import export_views

            if not os.path.exists(os.path.join(self.source_dir, "..", "..", "gtcuts.png")):
                cutvs = [list(vpair) for vpair in self.gt_cutvpairs]
                cylinderpos = ogvs[cutvs]
                cylindervals = np.ones((len(cylinderpos), 2)) # E x 2

                if len(cutvs) == 0:
                    cylinderpos = None
                    cylindervals = None

                export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                            filename=f"gtcuts.png",
                                plotname=f"Ground Truth Cuts", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals,
                                outline_width=0.01, cmap = plt.get_cmap('Reds'),
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

        # if self.args.gtjacobianloss:
        #     self.gt_jacobians = torch.load(os.path.join(self.source_dir, "..", "..", "gt_j.pt"))

        #     self.__loaded_data['gt_jacobians'] = self.gt_jacobians

        if self.args.gtedgeloss:
            # GT weights (need to be mapped to network output)
            with open(os.path.join(self.source_dir, "..", "..", "gt_cutvpairs.pkl"), 'rb') as f:
                self.gt_cutvpairs = pickle.load(f)

            # NOTE: network output is ordered by edge indexing
            self.gt_weights = torch.ones(len(self.edge_vpairs), device=device).double()

            # NOTE: sometimes we accidentally saved cut vpair indices that are on boundary of original mesh -- oops
            cut_es = [self.meshe_to_meshenobound[self.vpair_to_meshe[frozenset(vpair)]] for vpair in self.gt_cutvpairs if self.vpair_to_meshe[frozenset(vpair)] in self.meshe_to_meshenobound.keys()]
            self.gt_weights[cut_es] = 1e-8
            self.__loaded_data['gt_weights'] = self.gt_weights

            # Also plot GT cuts
            if not os.path.exists(os.path.join(self.source_dir, "..", "..", "gtcuts.png")):
                from results_saving_scripts.plot_uv import export_views
                import matplotlib.pyplot as plt

                cutvs = [list(vpair) for vpair in self.gt_cutvpairs]
                cylinderpos = ogvs[cutvs]
                cylindervals = np.ones((len(cylinderpos), 2)) # E x 2

                export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                            filename=f"gtcuts.png",
                                plotname=f"Ground Truth Cuts", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals,
                                outline_width=0.01, cmap = plt.get_cmap('Reds'),
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

        # GT edgecutloss requires edgefreqdict.pkl
        if self.args.stitchingloss is not None and "weightededgecutloss" in self.args.stitchingloss:
            import dill as pickle
            edgedictpath = os.path.join(self.source_dir, "..", "..", "edgefreqdict.pkl")
            if not os.path.exists(edgedictpath):
                raise ValueError(f"Edge frequency dictionary not found in {edgedictpath}!")
            with open(edgedictpath, 'rb') as f:
                self.edgefreqdict = pickle.load(f)

            # Map frequencies to edge weight
            freqs = self.edgefreqdict
            self.edgecutweights = torch.ones(len(self.edge_vpairs), device=device).double()
            freqvalues = torch.zeros(len(self.edge_vpairs), device=device).double()
            # import pdb
            # pdb.set_trace()
            totalfreq = freqs['total']
            for vpair, freq in freqs.items():
                if vpair == "total":
                    continue
                eidx = self.vpair_to_meshe[vpair]
                eidx_nobound = self.meshe_to_meshenobound[eidx]
                freqvalues[eidx_nobound] = freq / totalfreq
            self.edgecutweights -= freqvalues
            self.__loaded_data['edgecutweights'] = self.edgecutweights

        self.__loaded_data['valid_edge_pairs'] = self.valid_edge_pairs
        self.__loaded_data['facepairs'] = self.facepairs
        self.__loaded_data['edge_vpairs'] = self.edge_vpairs
        self.__loaded_data['elens_nobound'] = self.elens_nobound
        self.__loaded_data['ogedge_vpairs_nobound'] = self.ogedge_vpairs_nobound
        self.__loaded_data['facepairs_nobound'] = self.facepairs_nobound

        # Compute normals if normal loss
        if self.args.normalloss:
            # TODO: Building halfedge from scratch SLOW
            # TODO: replace below with boilerplate code using standard vs, fs
            from meshing.analysis import computeFaceNormals
            computeFaceNormals(mesh)
            self.fnormals = torch.from_numpy(mesh.fnormals).to(device)
            self.__loaded_data['fnormals'] = self.fnormals

            # If continue training, then load the normalgrid
            if self.args.continuetrain:
                ckptdir = os.path.join(self.args.outputdir, self.args.expname)
                for objname in os.listdir(ckptdir):
                    if not os.path.isdir(os.path.join(ckptdir, objname)):
                        continue
                    ngriddir = os.path.join(ckptdir, objname, "normalgrid.pt")
                    if os.path.exists(ngriddir):
                        normalgrid = torch.load(ngriddir)
                    else:
                        print(f"Could not find normalgrid in {ngriddir}. Resetting from scratch ... ")
                        normalgrid = torch.zeros((3, 256, 256), device=device, requires_grad=True, dtype=torch.float64)
            # Otherwise, initialize from scratch
            else:
                normalgrid = torch.zeros((3, 256, 256), device=device, requires_grad=True, dtype=torch.float64)
            self.normalgrid = normalgrid
            self.__loaded_data['normalgrid'] = normalgrid

        ### Texture Stuff ###
        if self.args.rasterizer is not None and "nvd" in self.args.rasterizer:
            # Setup nvdiffrast attributes
            pass
        elif self.args.rasterizer == "kaolin":
            # TODO: setup kaolin per-mesh attributes
            pass

        # Store texture names associated with this mesh
        if self.args.rasterizer is not None:
            assert os.path.exists(os.path.join(self.source_dir, "..", "..", "textures.txt"))

            # TODO: Replace with a new args that takes the data path
            with open(os.path.join(self.source_dir, "..", "..", "textures.txt"), 'r') as f:
                self.texturenames = [line.strip() for line in f.readlines()]

            if "sds" in self.args.visualloss:
                assert os.path.exists(os.path.join(self.source_dir, "..", "..", "texturedescrip.txt"))

                # Texture descriptions for SDS
                descripfile = os.path.join(self.source_dir, "..", "..", "texturedescrip.txt")

                texturedescrip = []
                with open(descripfile, 'r') as f:
                    for line in f:
                        texturedescrip.append(line.strip())

                assert len(texturedescrip) == len(self.texturenames)

                # HACK: Use single texture prompt for every texture
                self.texturedescrip = {}
                if len(self.args.sdscaption) > 0:
                    sdscaption = ' '.join(self.args.sdscaption)
                    for i in range(len(self.texturenames)):
                        self.texturedescrip[self.texturenames[i]] = sdscaption
                else:
                    for i in range(len(self.texturenames)):
                        self.texturedescrip[self.texturenames[i]] = texturedescrip[i]

            elif self.args.sdscheckpointing or self.args.clipcheckpointing:
                # Load texture descrip file if it exists anyway
                # Texture descriptions for SDS
                descripfile = os.path.join(self.source_dir, "..", "..", "texturedescrip.txt")

                if os.path.exists(descripfile):
                    texturedescrip = []
                    with open(descripfile, 'r') as f:
                        for line in f:
                            texturedescrip.append(line.strip())

                    assert len(texturedescrip) == len(self.texturenames)

                    self.texturedescrip = {}
                    if len(self.args.sdscaption) > 0:
                        sdscaption = ' '.join(self.args.sdscaption)
                        for i in range(len(self.texturenames)):
                            self.texturedescrip[self.texturenames[i]] = sdscaption
                    else:
                        for i in range(len(self.texturenames)):
                            self.texturedescrip[self.texturenames[i]] = texturedescrip[i]

        ### Initialize embeddings ###
        # keepidxs determines which edges to compute the loss over
        self.keepidxs = np.arange(len(self.edge_vpairs))

        # Load fixed vertices list if set
        self.fixedvs = None
        if self.args.keepuv and os.path.exists(os.path.join(self.source_dir, "..", "..", "fixedvs.npy")):
            self.fixedvs = np.load(os.path.join(self.source_dir, "..", "..", "fixedvs.npy"))

        # Precompute Tutte if set
        if self.init == "tutte":
            if os.path.exists(os.path.join(self.source_dir, "tuttefuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tutteuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttej.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "tuttetranslate.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "cutvs.npy")) and \
                os.path.exists(os.path.join(self.source_dir, "cutfs.npy")) and \
                    not new_init and self.args.ignorei == 0:
                self.tuttefuv = torch.load(os.path.join(self.source_dir, "tuttefuv.pt"))
                self.tutteuv = torch.load(os.path.join(self.source_dir, "tutteuv.pt"))
                self.tuttej = torch.load(os.path.join(self.source_dir, "tuttej.pt"))
                self.tuttetranslate = torch.load(os.path.join(self.source_dir, "tuttetranslate.pt"))
                self.cutvs = np.load(os.path.join(self.source_dir, "cutvs.npy"))
                self.cutfs = np.load(os.path.join(self.source_dir, "cutfs.npy"))

                if self.args.softpoisson or self.args.arch == "directedge":
                    self.initweights = torch.load(os.path.join(self.source_dir, f"tutteinitweights_{self.args.softpoisson}.pt"))
                    self.initcuts = torch.load(os.path.join(self.source_dir, f"tutteinitcuts_{self.args.softpoisson}.pt"))

                # Get delete idxs and remove from keepidxs
                if self.args.removecutfromloss:
                    deleteidxs = np.where(self.initweights < 0)[0]
                    self.keepidxs = np.delete(self.keepidxs, deleteidxs)
            else:
                from utils import tutte_embedding, get_local_tris, generate_random_cuts, generate_boundary_cut, make_cut

                ogsoup = ogvs[ogfs]
                cutvs = None
                rng = default_rng()
                n_cuts = rng.integers(self.args.min_cuts, self.args.max_cuts+1)

                ## ==== DEBUGGING: manually set some edges to cut in the initialization ====
                ignore_edges = [298, 464, 555, 301, 304, 605, 456, 46,717,552,700,699,692, 691,
                        647, 190, 16, 200, 761, 757, 342, 662, 577, 122, 510, 79, 20]
                ignoreset = ignore_edges[:self.args.ignorei]
                if len(ignoreset) > 0:
                    cutvs = []
                    for i in range(len(ignoreset)):
                        e = ignoreset[i]
                        twovs = [v.index for v in mesh.topology.edges[e].two_vertices()]
                        if i == 0:
                            if not mesh.topology.vertices[twovs[0]].onBoundary():
                                assert mesh.topology.vertices[twovs[1]].onBoundary()
                                twovs = twovs[::-1]
                            cutvs.extend(twovs)
                        else:
                            # One of the vertices should be same as the previous
                            if twovs[0] == cutvs[-1]:
                                cutvs.append(twovs[1])
                            elif twovs[1] == cutvs[-1]:
                                cutvs.append(twovs[0])
                            else:
                                raise ValueError(f"Vertex pair {twovs} does not share a vertex with the previous edge in cut set {cutvs}!")

                    cutvs = np.array(cutvs)
                    make_cut(mesh, cutvs)
                    set_new_tutte = True
                else:
                    if self.args.simplecut and n_cuts > 0:
                        cutvs = generate_boundary_cut(mesh, max_cuts = n_cuts)
                    elif n_cuts > 0:
                        cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)

                if cutvs is not None:
                    # Unit test: mesh is still connected
                    vs, fs, es = mesh.export_soup()
                    testsoup = PolygonSoup(vs, fs)
                    n_components = testsoup.nConnectedComponents()
                    assert n_components == 1, f"After cutting found {n_components} components!"

                    # Save new topology
                    self.cutvs = vs
                    self.cutfs = fs

                    ## Unit Test: Check that cutfs order is same as ogfs order (the soups should be the same)
                    cutsoup = vs[fs]
                    ogsoup = ogvs[ogfs]
                    np.testing.assert_close(cutsoup, ogsoup)

                    # Only replace Tutte if no nan
                    newtutte = torch.from_numpy(tutte_embedding(vs, fs)[0]).unsqueeze(0).type(self.__ttype) # 1 x V x 2
                    set_new_tutte = False
                    if torch.all(~torch.isnan(newtutte)):
                        self.tutteuv = newtutte

                        # Convert Tutte to 3-dim
                        self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        # souptutte = self.tutteuv.squeeze()[:,:2][self.cutfs,:].reshape(-1, 2)
                        # soupvs = torch.from_numpy(ogsoup.reshape(-1, 3)).to(device)
                        # soupfs = torch.from_numpy(np.arange(len(soupvs)).reshape(-1, 3)).to(device)
                        vs = torch.from_numpy(self.cutvs)
                        fs = torch.from_numpy(self.cutfs)
                        self.tuttej = get_jacobian_torch(vs, fs, self.tutteuv.squeeze()[:,:2], device=device) # F x 2 x 3
                        self.tuttej = torch.cat([self.tuttej, torch.zeros(self.tuttej.shape[0], 1, self.tuttej.shape[2])], dim=1)

                        # self.tuttej = get_jacobian_torch(torch.from_numpy(mesh.vertices), torch.from_numpy(mesh.faces), self.tutteuv.squeeze()[:,:2], device=device) # F x 2 x 3
                        # self.tuttej = torch.cat([self.tuttej, torch.zeros(self.tuttej.shape[0], 1, self.tuttej.shape[2])], dim=1)

                        if torch.any(~torch.isfinite(self.tuttej)):
                            print("Tutte Jacobians have NaNs!")
                        else:
                            set_new_tutte = True

                    # Otherwise, just use the default Tutte
                    if not set_new_tutte:
                        self.tutteuv = torch.from_numpy(tutte_embedding(vertices.detach().cpu().numpy(), faces)[0]).unsqueeze(0).type(self.__ttype) # 1 x V x 2

                        # Convert Tutte to 3-dim
                        self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        self.tuttej = get_jacobian_torch(vertices, faces, self.tutteuv.squeeze()[:,:2], device=device) # F x 2 x 3
                        self.tuttej = torch.cat([self.tuttej, torch.zeros(self.tuttej.shape[0], 1, self.tuttej.shape[2])], dim=1)

                        # Reset cut topo to original topo
                        self.cutvs = vertices.detach().cpu().numpy()
                        self.cutfs = faces
                    ### Set initialization weights here based on the cuts (mark cut edge weights to 0)
                    else:
                        # DEBUGGING: Check the laplacian in self.poisson (indexed by edge_vpairs)
                        # TODO: all indexes should be 0 in the soft poisson laplacian
                        cutvedges = [frozenset([cutvs[i], cutvs[i+1]]) for i in range(len(cutvs)-1)]
                        deleteidxs = []
                        for cutvpair in cutvedges:
                            eidx = self.vpair_to_meshe[cutvpair]
                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            deleteidxs.append(eidx_nobound)

                            if self.args.spweight == "sigmoid":
                                self.initweights[eidx_nobound] = -10
                            elif self.args.spweight == ["softmax", "nonzero"]:
                                self.initweights[eidx_nobound] = 1e-7
                            elif self.args.spweight in ["seamless", "cosine"]:
                                self.initweights[eidx_nobound] = -0.5

                            self.initcuts[eidx_nobound] = 1

                        if self.args.removecutfromloss:
                            self.keepidxs = np.delete(self.keepidxs, deleteidxs)

                        # Plot init cuts
                        from results_saving_scripts.plot_uv import export_views
                        cutvedges = [list(vpair) for vpair in cutvedges]
                        cylinderpos = ogvs[cutvedges]
                        cylindervals = np.ones((len(cylinderpos), 2)) # E x 2
                        export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                                    filename=f"initcuts.png",
                                        plotname=f"Initial Cuts", cylinders=cylinderpos,
                                        cylinder_scalars=cylindervals,
                                        outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                                        device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=1, shading=False)
                else:
                    self.tutteuv = torch.from_numpy(tutte_embedding(vertices.detach().cpu().numpy(), faces)[0]).unsqueeze(0).type(self.__ttype) # 1 x V x 2

                    # Get Jacobians
                    souptutte = self.tutteuv.squeeze()[:,:2][self.cutfs, :].reshape(-1, 2)
                    soupvs = torch.from_numpy(ogsoup.reshape(-1, 3))
                    soupfs = torch.arange(len(soupvs)).reshape(-1, 3)
                    self.tuttej = get_jacobian_torch(soupvs, soupfs, souptutte, device=device) # F x 2 x 3
                    self.tuttej = torch.cat([self.tuttej, torch.zeros(self.tuttej.shape[0], 1, self.tuttej.shape[2])], dim=1)

                    # Convert Tutte to 3-dim
                    self.tutteuv = torch.cat([self.tutteuv, torch.zeros(self.tutteuv.shape[0], self.tutteuv.shape[1], 1)], dim=-1)

                    # Reset cut topo to original topo
                    self.cutvs = vertices.detach().cpu().numpy()
                    self.cutfs = faces

                # DEBUG: make sure we can get back the original UVs up to global translation
                # NOTE: We compare triangle centroids bc face indexing gets messed up after cutting
                fverts = torch.from_numpy(ogvs[ogfs])
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.tuttej[:,:2,:].transpose(2,1)))

                if new_init and self.init == "tutte" and set_new_tutte:
                    checktutte = self.tutteuv[0,fs,:2]
                    self.tuttefuv = self.tutteuv[:,fs,:2] # B x F x 3 x 2
                else:
                    checktutte = self.tutteuv[0,faces,:2]
                    self.tuttefuv = self.tutteuv[:,faces,:2] # B x F x 3 x 2

                # diff = pred_V - checktutte
                # diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle global translation
                # torch.testing.assert_close(diff.float(), torch.zeros_like(diff), rtol=float('inf'), atol=1e-5)

                ## Save the global translations
                self.tuttetranslate = (checktutte - pred_V)[:,:,:2]

                # Cache everything (only if not continuous new init)
                if not new_init and self.args.ignorei == 0:
                    torch.save(self.tuttefuv, os.path.join(self.source_dir, "tuttefuv.pt"))
                    torch.save(self.tutteuv, os.path.join(self.source_dir, "tutteuv.pt"))
                    torch.save(self.tuttej, os.path.join(self.source_dir, "tuttej.pt"))
                    torch.save(self.tuttetranslate, os.path.join(self.source_dir, "tuttetranslate.pt"))

                    if self.args.softpoisson or self.args.arch == 'directedge':
                        torch.save(self.initweights, os.path.join(self.source_dir, f"tutteinitweights_{self.args.softpoisson}.pt"))
                        torch.save(self.initcuts, os.path.join(self.source_dir, f"tutteinitcuts_{self.args.softpoisson}.pt"))

                    np.save(os.path.join(self.source_dir, f"cutvs.npy"), self.cutvs)
                    np.save(os.path.join(self.source_dir, f"cutfs.npy"), self.cutfs)

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['tuttefuv'] = self.tuttefuv
            self.__loaded_data['tutteuv'] = self.tutteuv
            self.__loaded_data['tuttej'] = self.tuttej
            self.__loaded_data['tuttetranslate'] = self.tuttetranslate

            if self.args.softpoisson or self.args.arch == 'directedge':
                self.__loaded_data['initweights'] = self.initweights

            if self.initjinput:
                # NOTE: If vertex features, then just use the initial UV position
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Map face Jacobians to vertices by aggregating all Jacobians for each incident vertex
                    vertsj = []
                    for v in range(len(vertices)):
                        vertfs = np.where(faces == v)[0]
                        vertj = torch.mean(self.tuttej[vertfs], dim=0)
                        vertsj.append(vertj.flatten()) # 6-dim
                    vertsj = torch.stack(vertsj, dim=0) # V x 6
                    self.input_features = torch.cat([self.input_features, vertsj], dim=1)
                elif self.args.arch == "mlp":
                    self.input_features = torch.cat([self.input_features, self.prej.reshape(-1, 6)], dim=1)

            if self.args.initweightinput:
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Each vertex gets average of incident edge weights
                    vertsw = []
                    for v in range(len(vertices)):
                        vertes = torch.where(self.ogedge_vpairs_nobound == v)[0]

                        # All adjacent edges to vertex are cut
                        if len(vertes) == 0:
                            vertsw.append(torch.zeros(1, device=device))
                        else:
                            vertw = torch.mean(self.initcuts[vertes])
                            vertsw.append(vertw) # scalar

                    vertsw = torch.tensor(vertsw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, vertsw], dim=1)
                elif self.args.arch == "mlp":
                    # Each face gets average of incident edge weights
                    facesw = []
                    for face in faces:
                        tmp = []
                        for i in range(3):
                            e = frozenset([face[i], face[(i+1)%3]])
                            eidx = self.vpair_to_meshe[e]
                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            tmp.append(self.initcuts[eidx_nobound])
                        facesw.append(torch.mean(tmp))
                    facesw = torch.tensor(facesw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, facesw], dim=1)

        elif self.init == "precut":
            if os.path.exists(os.path.join(self.source_dir, "prefuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "preuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "prej.pt")) and \
                os.path.exists(os.path.join(self.source_dir, f"preinitweights_{self.args.spweight}.pt")) and \
                os.path.exists(os.path.join(self.source_dir, f"preinitcuts.pt")) and \
                    not self.args.overwriteinit:

                self.prefuv = torch.load(os.path.join(self.source_dir, "prefuv.pt"))
                self.preuv = torch.load(os.path.join(self.source_dir, "preuv.pt"))
                self.prej = torch.load(os.path.join(self.source_dir, "prej.pt"))
                self.initweights = torch.load(os.path.join(self.source_dir, f"preinitweights_{self.args.spweight}.pt"))
                self.initcuts = torch.load(os.path.join(self.source_dir, f"preinitcuts.pt"))

                # Get delete idxs and remove from keepidxs
                if self.args.removecutfromloss:
                    deleteidxs = np.where(self.initweights < 0)[0]
                    self.keepidxs = np.delete(self.keepidxs, deleteidxs)

                # Plot init cuts if haven't yet
                if not os.path.exists(os.path.join(self.source_dir, "..", "..", "initcuts.png")):
                    initcuts = pickle.load(open(os.path.join(self.source_dir, "..", "..", "initialcuts.pkl"), "rb"))
                    from results_saving_scripts.plot_uv import export_views
                    initcuts = [list(cut) for cut in initcuts]
                    cylinderpos = ogvs[initcuts]
                    cylindervals = np.ones((len(cylinderpos), 2)) # E x 2
                    export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                                filename=f"initcuts.png",
                                    plotname=f"Initial Cuts", cylinders=cylinderpos,
                                    cylinder_scalars=cylindervals,
                                    outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                                    device="cpu", n_sample=30, width=200, height=200,
                                    vmin=0, vmax=1, shading=False)

                # Plot init uv if haven't yet
                if not os.path.exists(os.path.join(self.source_dir, "..", "..", "inituv.png")):

                    # Plot GT UVs (for visualization)
                    from matplotlib.tri import Triangulation
                    import matplotlib.pyplot as plt

                    inituvs = self.prefuv.reshape(-1, 2).detach().cpu().numpy()
                    triangles = np.arange(len(inituvs)).reshape(-1, 3)
                    tris = Triangulation(inituvs[:, 0], inituvs[:, 1], triangles=triangles)
                    fig, axs = plt.subplots(figsize=(5, 5))
                    axs.set_title("Init UVs")
                    cmap = plt.get_cmap("tab20")
                    axs.tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                                        linewidth=0.1, edgecolor="black")
                    plt.axis('off')
                    axs.axis('equal')
                    plt.savefig(os.path.join(self.source_dir, "..", "..", "inituv.png"))
                    plt.close(fig)
                    plt.cla()
            else:
                from utils import get_local_tris

                device = vertices.device

                # We load the cutmesh along with the cut vpairs to initialize the pre embedding
                cutsoup = PolygonSoup.from_obj(os.path.join(self.source_dir, "..", "..", f"{self.source_ind}.obj"))
                initcuts = pickle.load(open(os.path.join(self.source_dir, "..", "..", "initialcuts.pkl"), "rb"))

                fverts = vertices[faces]

                self.preuv = torch.from_numpy(cutsoup.uvs).unsqueeze(0).type(self.__ttype) # 1 x V x 2

                # Normalize if set
                if self.args.normalizeinit:
                    self.preuv -= torch.mean(self.preuv, dim=1, keepdim=True)
                    self.preuv /= (2 * torch.max(torch.linalg.norm(self.preuv, dim=2)))

                # Convert everything to soup indexing for Jacobian calculation
                preuv_soup = self.preuv[:, cutsoup.face_uv.astype(int)].reshape(1, -1, 2)

                # Get Jacobians
                soupvs = fverts.reshape(-1, 3).to(device)
                soupfs = torch.arange(len(soupvs), device=device).reshape(len(faces), 3)
                self.prej = get_jacobian_torch(soupvs, soupfs, preuv_soup.squeeze()[:,:2], device=device).double() # F x 2 x 3
                self.prej = torch.cat([self.prej, torch.zeros(self.prej.shape[0], 1, self.prej.shape[2])], dim=1)

                deleteidxs = []
                for cutvpair in initcuts:
                    eidx = self.vpair_to_meshe[cutvpair]
                    if eidx not in self.meshe_to_meshenobound.keys(): continue
                    eidx_nobound = self.meshe_to_meshenobound[eidx]
                    deleteidxs.append(eidx_nobound)

                    if self.args.spweight == "sigmoid":
                        self.initweights[eidx_nobound] = -10
                    elif self.args.spweight == ["softmax", "nonzero"]:
                        self.initweights[eidx_nobound] = 1e-7
                    elif self.args.spweight in ["seamless", "cosine"]:
                        self.initweights[eidx_nobound] = -0.5

                    self.initcuts[eidx_nobound] = 1

                    if self.args.removecutfromloss:
                        self.keepidxs = np.delete(self.keepidxs, deleteidxs)

                # Plot init cuts
                from results_saving_scripts.plot_uv import export_views
                initcuts = [list(cut) for cut in initcuts]
                cylinderpos = ogvs[initcuts]
                cylindervals = np.ones((len(cylinderpos), 2)) # E x 2

                if len(initcuts) == 0:
                    cylinderpos = None
                    cylindervals = None

                export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                            filename=f"initcuts.png",
                                plotname=f"Initial Cuts", cylinders=cylinderpos,
                                cylinder_scalars=cylindervals,
                                outline_width=0.01, cmap = plt.get_cmap('Reds_r'),
                                device="cpu", n_sample=30, width=200, height=200,
                                vmin=0, vmax=1, shading=False)

                # Unit Test: jacobians recover pre uvs
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.prej[:,:2,:].transpose(2,1)))

                checkpre = self.preuv[0,cutsoup.face_uv.astype(int),:2]
                self.prefuv = self.preuv[:,cutsoup.face_uv.astype(int),:2] # B x F x 3 x 2

                # Account for fgroups
                if self.fgroups is not None:
                    for fgroup in np.unique(self.fgroups):
                        fgroupmask = self.fgroups == fgroup
                        diff = pred_V[fgroupmask] - checkpre[fgroupmask]
                        diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                        torch.testing.assert_close(torch.zeros(diff.shape).double(), diff, rtol=float('inf'), atol=1e-2)
                else:
                    diff = pred_V - checkpre
                    diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                    torch.testing.assert_close(torch.zeros(diff.shape).double(), diff, rtol=float('inf'), atol=1e-2)

                # Cache everything
                torch.save(self.prefuv, os.path.join(self.source_dir, "prefuv.pt"))
                torch.save(self.preuv, os.path.join(self.source_dir, "preuv.pt"))
                torch.save(self.prej, os.path.join(self.source_dir, "prej.pt"))
                torch.save(self.initweights, os.path.join(self.source_dir, f"preinitweights_{self.args.spweight}.pt"))
                torch.save(self.initcuts, os.path.join(self.source_dir, f"preinitcuts.pt"))

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['prefuv'] = self.prefuv
            self.__loaded_data['preuv'] = self.preuv
            self.__loaded_data['prej'] = self.prej

            if self.args.softpoisson or self.args.arch == 'directedge':
                self.__loaded_data['initweights'] = self.initweights

            if self.initjinput:
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Map face Jacobians to vertices by aggregating all Jacobians for each incident vertex
                    vertsj = []
                    if self.initvs is not None and self.initfs is not None:
                        for v in range(len(self.initvs)):
                            vertfs = torch.where(self.initfs == v)[0]
                            vertj = torch.mean(self.prej[vertfs], dim=0)
                            vertsj.append(vertj.flatten()) # 6-dim
                    else:
                        for v in range(len(vertices)):
                            vertfs = np.where(faces == v)[0]
                            vertj = torch.mean(self.prej[vertfs], dim=0)
                            vertsj.append(vertj.flatten()) # 6-dim
                    vertsj = torch.stack(vertsj, dim=0) # V x 6
                    self.input_features = torch.cat([self.input_features, vertsj], dim=1)
                elif self.args.arch == "mlp":
                    self.input_features = torch.cat([self.input_features, self.prej[:, :2].reshape(-1, 6)], dim=1)

            if self.args.initweightinput:
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Each vertex gets average of incident edge weights
                    vertsw = []
                    if self.initvs is not None and self.initfs is not None:
                        vertsw = torch.zeros(len(self.initvs), device=device).double()
                        for v in range(len(vertices)):
                            vertes = torch.where(self.ogedge_vpairs_nobound == v)[0]
                            vlocs = np.where(self.fuv_to_v == v)[0]
                            # All adjacent edges to vertex are cut
                            if len(vertes) == 0:
                                vertsw[vlocs] = torch.zeros(len(vlocs), device=device)
                            else:
                                vertw = torch.mean(self.initcuts[vertes])
                                vertsw[vlocs] = vertw # scalar
                    else:
                        for v in range(len(vertices)):
                            vertes = torch.where(self.ogedge_vpairs_nobound == v)[0]
                            # All adjacent edges to vertex are cut
                            if len(vertes) == 0:
                                vertsw.append(torch.zeros(1, device=device))
                            else:
                                vertw = torch.mean(self.initcuts[vertes])
                                vertsw.append(vertw) # scalar
                    vertsw = torch.tensor(vertsw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, vertsw], dim=1)
                elif self.args.arch == "mlp":
                    # Each face gets average of incident edge weights
                    facesw = []
                    for face in faces:
                        tmp = []
                        for i in range(3):
                            e = frozenset([face[i].item(), face[(i+1)%3].item()])
                            eidx = self.vpair_to_meshe[e]

                            # Ignore the boundaries
                            if eidx not in self.meshe_to_meshenobound.keys(): continue

                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            tmp.append(self.initcuts[eidx_nobound])
                        facesw.append(np.mean(tmp))
                    facesw = torch.tensor(facesw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, facesw], dim=1)

        elif self.init == "slim":
            if os.path.exists(os.path.join(self.source_dir, "slimfuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimuv.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimj.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "slimtranslate.pt")) and \
                os.path.exists(os.path.join(self.source_dir, f"sliminitweights_{self.args.softpoisson}.pt")) and \
                os.path.exists(os.path.join(self.source_dir, f"sliminitcuts_{self.args.softpoisson}.pt")) and \
                os.path.exists(os.path.join(self.source_dir, "cutvs.npy")) and \
                os.path.exists(os.path.join(self.source_dir, "cutfs.npy")) and \
                    not new_init and self.args.ignorei == 0:

                self.slimfuv = torch.load(os.path.join(self.source_dir, "slimfuv.pt"))
                self.slimuv = torch.load(os.path.join(self.source_dir, "slimuv.pt"))
                self.slimj = torch.load(os.path.join(self.source_dir, "slimj.pt"))
                self.slimtranslate = torch.load(os.path.join(self.source_dir, "slimtranslate.pt"))
                self.initweights = torch.load(os.path.join(self.source_dir, f"sliminitweights_{self.args.softpoisson}.pt"))
                self.cutvs = np.load(os.path.join(self.source_dir, "cutvs.npy"))
                self.cutfs = np.load(os.path.join(self.source_dir, "cutfs.npy"))
                self.initcuts = torch.load(os.path.join(self.source_dir, f"sliminitcuts_{self.args.softpoisson}.pt"))

                # Get delete idxs and remove from keepidxs
                if self.args.removecutfromloss:
                    deleteidxs = np.where(self.initweights < 0)[0]
                    self.keepidxs = np.delete(self.keepidxs, deleteidxs)
            else:
                from utils import SLIM, get_local_tris, generate_random_cuts, generate_boundary_cut

                device = vertices.device
                fverts = vertices[faces]

                rng = default_rng()
                n_cuts = rng.integers(self.args.min_cuts, self.args.max_cuts+1)
                cutvs = None
                if self.args.simplecut and n_cuts > 0:
                    cutvs = generate_boundary_cut(mesh, max_cuts = n_cuts)
                else:
                    cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)

                # Unit test: mesh is still connected
                if cutvs is not None:
                    vs, fs, es = mesh.export_soup()
                    testsoup = PolygonSoup(vs, fs)
                    n_components = testsoup.nConnectedComponents()
                    assert n_components == 1, f"After cutting found {n_components} components!"

                    # Save new topology
                    self.cutvs = vs
                    self.cutfs = fs

                    cutsoup = vs[fs]
                    ogsoup = ogvs[ogfs]
                    np.testing.assert_close(cutsoup, ogsoup)

                    # Only replace SLIM if nan
                    newslim = torch.from_numpy(SLIM(mesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2
                    set_new_slim = False
                    if torch.all(~torch.isnan(newslim)):
                        self.slimuv = newslim

                        # Convert slim to 3-dim
                        self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        self.slimj = get_jacobian_torch(torch.from_numpy(self.cutvs), torch.from_numpy(self.cutfs), self.slimuv.squeeze()[:,:2], device=device) # F x 2 x 3
                        self.slimj = torch.cat([self.slimj, torch.zeros(self.slimj.shape[0], 1, self.slimj.shape[2])], dim=1)

                        if torch.any(~torch.isfinite(self.slimj)):
                            print("SLIM Jacobians have NaNs!")
                        else:
                            set_new_slim = True

                    # Otherwise, just use SLIM with no cutting
                    if not set_new_slim:
                        self.slimuv = torch.from_numpy(SLIM(mesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2

                        # Convert slim to 3-dim
                        self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                        # Get Jacobians
                        self.slimj = get_jacobian_torch(vertices, faces, self.slimuv.squeeze()[:,:2], device=device) # F x 2 x 3
                        self.slimj = torch.cat([self.slimj, torch.zeros(self.slimj.shape[0], 1, self.slimj.shape[2])], dim=1)

                        # Reset cut topo to original topo
                        self.cutvs = vertices.detach().cpu().numpy()
                        self.cutfs = faces.detach().numpy()
                    else:
                        cutvedges = [frozenset([cutvs[i], cutvs[i+1]]) for i in range(len(cutvs)-1)]
                        deleteidxs = []
                        for cutvpair in cutvedges:
                            eidx = self.vpair_to_meshe[cutvpair]
                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            deleteidxs.append(eidx_nobound)

                            if self.args.spweight == "sigmoid":
                                self.initweights[eidx_nobound] = -10
                            elif self.args.spweight == ["softmax", "nonzero"]:
                                self.initweights[eidx_nobound] = 1e-7
                            elif self.args.spweight in ["seamless", "cosine"]:
                                self.initweights[eidx_nobound] = -0.5

                            self.initcuts[eidx_nobound] = 1

                        if self.args.removecutfromloss:
                            self.keepidxs = np.delete(self.keepidxs, deleteidxs)

                        # Plot init cuts
                        from results_saving_scripts.plot_uv import export_views
                        cutvedges = [list(vpair) for vpair in cutvedges]
                        cylinderpos = ogvs[cutvedges]
                        cylindervals = np.ones((len(cylinderpos), 2)) # E x 2
                        export_views(ogvs, ogfs, os.path.join(self.source_dir, "..", ".."),
                                    filename=f"initcuts.png",
                                        plotname=f"Initial Cuts", cylinders=cylinderpos,
                                        cylinder_scalars=cylindervals,
                                        outline_width=0.01, cmap = plt.get_cmap('Reds'),
                                        device="cpu", n_sample=30, width=200, height=200,
                                        vmin=0, vmax=1, shading=False)
                else:
                    self.slimuv = torch.from_numpy(SLIM(mesh, iters=self.args.slimiters)[0]).unsqueeze(0) # 1 x V x 2

                    # Get Jacobians
                    self.slimj = get_jacobian_torch(vertices, faces, self.slimuv.squeeze()[:,:2], device=device) # F x 2 x 3
                    self.slimj = torch.cat([self.slimj, torch.zeros(self.slimj.shape[0], 1, self.slimj.shape[2])], dim=1)

                    # Convert slim to 3-dim
                    self.slimuv = torch.cat([self.slimuv, torch.zeros(self.slimuv.shape[0], self.slimuv.shape[1], 1)], dim=-1)

                    # Reset cut topo to original topo
                    self.cutvs = vertices.detach().cpu().numpy()
                    self.cutfs = faces.detach().numpy()

                # DEBUG: make sure we can get back the original UVs up to global translation
                # NOTE: We compare triangle centroids bc face indexing gets messed up after cutting
                fverts = vertices[faces]
                # pred_V = torch.einsum("abc,acd->abd", (self.slimj[0,:,:2,:], fverts)).transpose(1,2)
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.slimj[:,:2,:].transpose(2,1)))

                if new_init and self.init == "slim" and set_new_slim:
                    checkslim = self.slimuv[0,fs,:2]
                    self.slimfuv = self.slimuv[:,fs,:2] # B x F x 3 x 2
                else:
                    checkslim = self.slimuv[0,faces,:2]
                    self.slimfuv = self.slimuv[:,faces,:2] # B x F x 3 x 2

                # diff = pred_V - checkslim
                # diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
                # torch.testing.assert_close(diff.float(), torch.zeros(diff.shape), rtol=float('inf'), atol=1e-2)

                ## Save the global translations
                self.slimtranslate = (checkslim - pred_V)[:,:,:2]

                # Cache everything
                if not new_init:
                    torch.save(self.slimfuv, os.path.join(self.source_dir, "slimfuv.pt"))
                    torch.save(self.slimuv, os.path.join(self.source_dir, "slimuv.pt"))
                    torch.save(self.slimj, os.path.join(self.source_dir, "slimj.pt"))
                    torch.save(self.slimtranslate, os.path.join(self.source_dir, "slimtranslate.pt"))
                    torch.save(self.initweights, os.path.join(self.source_dir, f"sliminitweights_{self.args.softpoisson}.pt"))
                    torch.save(self.initcuts, os.path.join(self.source_dir, f"sliminitcuts_{self.args.softpoisson}.pt"))

                    np.save(os.path.join(self.source_dir, f"cutvs.npy"), self.cutvs)
                    np.save(os.path.join(self.source_dir, f"cutfs.npy"), self.cutfs)

            ## Store in loaded data so it gets mapped to device
            # Remove extraneous dimension
            self.__loaded_data['slimfuv'] = self.slimfuv
            self.__loaded_data['slimuv'] = self.slimuv
            self.__loaded_data['slimj'] = self.slimj
            self.__loaded_data['slimtranslate'] = self.slimtranslate

            if self.args.softpoisson or self.args.arch == "directedge":
                self.__loaded_data['initweights'] = self.initweights

            if self.initjinput:
                if self.args.arch == "diffusionnet":
                    # Map face Jacobians to vertices by aggregating all Jacobians for each incident vertex
                    vertsj = []
                    for v in range(len(vertices)):
                        vertfs = np.where(faces == v)[0]
                        vertj = torch.mean(self.slimj[vertfs], dim=0)
                        vertsj.append(vertj.flatten()) # 6-dim
                    vertsj = torch.stack(vertsj, dim=0) # V x 6
                    self.input_features = torch.cat([self.input_features, vertsj], dim=1)
                elif self.args.arch == "mlp":
                    self.input_features = torch.cat([self.input_features, self.prej.reshape(-1, 6)], dim=1)

            if self.args.initweightinput:
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Each vertex gets average of incident edge weights
                    vertsw = []
                    for v in range(len(vertices)):
                        vertes = torch.where(self.ogedge_vpairs_nobound == v)[0]
                        # All adjacent edges to vertex are cut
                        if len(vertes) == 0:
                            vertsw.append(torch.zeros(1, device=device))
                        else:
                            vertw = torch.mean(self.initcuts[vertes])
                            vertsw.append(vertw) # scalar
                    vertsw = torch.tensor(vertsw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, vertsw], dim=1)
                elif self.args.arch == "mlp":
                    # Each face gets average of incident edge weights
                    facesw = []
                    for face in faces:
                        tmp = []
                        for i in range(3):
                            e = frozenset([face[i], face[(i+1)%3]])
                            eidx = self.vpair_to_meshe[e]
                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            tmp.append(self.initcuts[eidx_nobound])
                        facesw.append(torch.mean(tmp))
                    facesw = torch.tensor(facesw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, facesw], dim=1)

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

                device = vertices.device
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

                self.isofuv = local_tris

                # Unit testing: face areas should be same as in 3D
                if self.args.debug:
                    from meshing.mesh import Mesh
                    from meshing.analysis import computeFaceAreas

                    mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().numpy())
                    computeFaceAreas(mesh)
                    fareas3d = mesh.fareas
                    fareas2d = 0.5 * np.abs(torch.linalg.det(torch.cat([torch.ones((len(self.isofuv), 1, 3)), self.isofuv.transpose(2,1)], dim=1)).numpy())

                    np.testing.assert_close(fareas3d, fareas2d, err_msg="Isometric embedding: all triangle areas should be same!")

                # Get jacobians
                # NOTE: For isometric init, vs/fs need to be based on triangle
                soupvs = fverts.reshape(-1, 3)
                soupfs = torch.arange(len(soupvs)).reshape(-1, 3).long().to(fverts.device)
                self.isoj = get_jacobian_torch(soupvs, soupfs, self.isofuv.reshape(-1, 2), device=device) # F x 2 x 3

                ## Debugging: make sure we can get back the original UVs up to global translation
                pred_V = torch.einsum("abc,acd->abd", (fverts, self.isoj.transpose(2,1)))
                diff = pred_V - self.isofuv
                diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle global translation
                torch.testing.assert_close(torch.zeros(diff.shape).double(), diff, rtol=float('inf'), atol=1e-2)

                ## Save the global translations
                self.isotranslate = self.isofuv - pred_V

                # Cache everything
                if not new_init:
                    torch.save(self.isofuv, os.path.join(self.source_dir, "isofuv.pt"))
                    torch.save(self.isoj, os.path.join(self.source_dir, "isoj.pt"))
                    torch.save(self.isotranslate, os.path.join(self.source_dir, "isotranslate.pt"))

            ## Ignore from loss
            ignore_edges = [298, 464, 555, 301, 304, 605, 456, 46,717,552,700,699,692, 691,
                    647, 190, 16, 200, 761, 757, 342, 662, 577, 122, 510, 79, 20]
            ignoreset = ignore_edges[:self.args.ignorei]
            if len(ignoreset) > 0:
                cutvs = []
                for i in range(len(ignoreset)):
                    e = ignoreset[i]
                    twovs = [v.index for v in mesh.topology.edges[e].two_vertices()]
                    if i == 0:
                        if not mesh.topology.vertices[twovs[0]].onBoundary():
                            assert mesh.topology.vertices[twovs[1]].onBoundary()
                            twovs = twovs[::-1]
                        cutvs.extend(twovs)
                    else:
                        # One of the vertices should be same as the previous
                        if twovs[0] == cutvs[-1]:
                            cutvs.append(twovs[1])
                        elif twovs[1] == cutvs[-1]:
                            cutvs.append(twovs[0])
                        else:
                            raise ValueError(f"Vertex pair {twovs} does not share a vertex with the previous edge in cut set {cutvs}!")

                cutvs = np.array(cutvs)
                cutvedges = [frozenset([cutvs[i], cutvs[i+1]]) for i in range(len(cutvs)-1)]
                deleteidxs = []
                for cutvpair in cutvedges:
                    eidx = self.vpair_to_meshe[cutvpair]
                    eidx_nobound = self.meshe_to_meshenobound[eidx]
                    deleteidxs.append(eidx_nobound)

                if self.args.removecutfromloss:
                    self.keepidxs = np.delete(self.keepidxs, deleteidxs)

            # fverts = ogvs[ogfs].reshape(-1, 3)
            # self.cutfs = np.arange(len(fverts)).reshape(-1, 3)

            ## Store in loaded data so it gets mapped to device
            # NOTE: need to transpose isoj to interpret as 2x3
            self.__loaded_data['isofuv'] = self.isofuv
            self.__loaded_data['isoj'] = self.isoj
            self.__loaded_data['isotranslate'] = self.isotranslate

            if self.initjinput:
                if self.args.arch == "diffusionnet":
                    # Map face Jacobians to vertices by aggregating all Jacobians for each incident vertex
                    vertsj = []
                    for v in range(len(vertices)):
                        vertfs = np.where(faces == v)[0]
                        vertj = torch.mean(self.isoj[vertfs], dim=0)
                        vertsj.append(vertj.flatten()) # 6-dim
                    vertsj = torch.stack(vertsj, dim=0) # V x 6
                    self.input_features = torch.cat([self.input_features, vertsj], dim=1)
                elif self.args.arch == "mlp":
                    self.input_features = torch.cat([self.input_features, self.prej.reshape(-1, 6)], dim=1)

            if self.args.initweightinput:
                if self.args.arch in ["diffusionnet", 'fullconv', 'fullconv2']:
                    # Each vertex gets average of incident edge weights
                    vertsw = []
                    for v in range(len(vertices)):
                        vertes = torch.where(self.ogedge_vpairs_nobound == v)[0]

                        # All adjacent edges to vertex are cut
                        if len(vertes) == 0:
                            vertsw.append(torch.zeros(1, device=device))
                        else:
                            vertw = torch.mean(self.initweights[vertes])
                            vertsw.append(vertw) # scalar

                    vertsw = torch.tensor(vertsw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, vertsw], dim=1)
                elif self.args.arch == "mlp":
                    # Each face gets average of incident edge weights
                    facesw = []
                    for face in faces:
                        tmp = []
                        for i in range(3):
                            e = frozenset([face[i], face[(i+1)%3]])
                            eidx = self.vpair_to_meshe[e]
                            eidx_nobound = self.meshe_to_meshenobound[eidx]
                            tmp.append(self.initcuts[eidx_nobound])
                        facesw.append(torch.mean(tmp))
                    facesw = torch.tensor(facesw).unsqueeze(1)
                    self.input_features = torch.cat([self.input_features, facesw], dim=1)

        ### Input features must have no nans ###
        assert not torch.any(torch.isnan(self.input_features)), "Input features have NaNs!"

        ### Dense: Use initialization jacobians as input
        if self.flatten == "input":
            if self.init == "tutte":
                self.flat_vector = self.tuttej.reshape(1, -1)
            elif self.init == "isometric":
                self.flat_vector = torch.cat([self.isoj, torch.zeros((self.isoj.shape[0], 1, 3))], dim=1).reshape(1, -1)
            elif self.init == "slim":
                self.flat_vector = self.slimj.reshape(1, -1)
            # nchannels = self.input_features.shape[1]
            # gsize = int(np.ceil(nchannels/9))
            # newchannels = []
            # for i in range(9):
            #     newchannels.append(torch.sum(self.input_features[:,i*gsize:(i+1)*gsize], dim=1))
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

    def load(self, source_v=None, source_f=None, new_init=False):
        if source_v is not None and source_f is not None:
            self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_array(source_v,source_f, self.source_dir, self.__ttype,
                                                                                       cpuonly=self.cpuonly, load_wks_samples=self.__use_wks,
                                                                                       load_wks_centroids=self.__use_wks,
                                                                                       top_k_eig=self.top_k_eig,
                                                                                       softpoisson=self.args.softpoisson,
                                                                                        sparse=self.args.sparsepoisson)
        else:
            if os.path.isdir(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_directory(self.source_dir, self.__ttype,
                                                                                               cpuonly=self.cpuonly,
                                                                                               load_wks_samples=self.__use_wks,
                                                                                               load_wks_centroids=self.__use_wks,
                                                                                               top_k_eig=self.top_k_eig,
                                                                                                softpoisson=self.args.softpoisson,
                                                                                                sparse=self.args.sparsepoisson)
            elif os.path.exists(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_file(self.source_dir, self.__ttype,
                                                                                          cpuonly=self.cpuonly,
                                                                                          load_wks_samples=self.__use_wks,
                                                                                          load_wks_centroids=self.__use_wks,
                                                                                          top_k_eig=self.top_k_eig,
                                                                                            softpoisson=self.args.softpoisson,
                                                                                            sparse=self.args.sparsepoisson)
            else:
                raise ValueError(f"Source dir path {self.source_dir} doesn't exist!")

        self.__init_from_mesh_data(new_init)

    def get_point_dim(self):
        # if self.flatten:
        #     return self.flat_vector.shape[1]
        return self.input_features.shape[1]

    def get_input_features(self):
        return self.input_features

    def get_mesh_centroid(self):
        return self.source_mesh_centroid

    def pin_memory(self):
        # self.poisson.pin_memory()
        # self.input_features.pin_memory()
        # self.source_vertices.pin_memory()
        # for key in self.__loaded_data.keys():
        #     self.__loaded_data[key].pin_memory()
        return self

    ##### Extra stuff for meshconv!!
    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]

        for v in vs:
            self.ve[v].remove(edge_id)

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def init_history(self):
        from source_njf.models import MeshUnion

        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.edges_count, dtype=np.int32),
                               'current2old': np.arange(self.edges_count, dtype=np.int32),
                               'edges_mask': [torch.ones(self.edges_count,dtype=torch.bool)],
                               'edges_count': [self.edges_count],
                               'edges': [self.edges.copy()],
                              }

        self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
            self.history_data['edges'].append(self.edges.copy())

    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.history_data['edges'].pop()
        self.history_data['edges_count'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.edges = self.history_data['edges'][-1]
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas

    def export(self, file=None, vcolor=None):
        if file is None:
            filename = self.source_ind
            file = '%s/%s_%d%s' % (self.source_dir, filename, self.pool_count, ".obj")

        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face