# InteractiveSegmentation: Standardized set of tests for parameterization loss functions
import numpy as np
import torch
from source_njf.utils import center_inplace, matmul_inplace

def get_local_tris(vertices, faces, device=torch.device("cpu")):
    fverts = vertices[faces].to(device)
    e1 = fverts[:, 1, :] - fverts[:, 0, :]
    e2 = fverts[:, 2, :] - fverts[:, 0, :]
    s = torch.linalg.norm(e1, dim=1)
    t = torch.linalg.norm(e2, dim=1)
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1))
    x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)])
    y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)])
    local_tris = torch.stack((x, y), dim=-1).reshape(len(angle), 3, 2)
    return local_tris

def meshArea2D(vertices, faces, return_fareas = False):
    # Get edge vectors
    fverts = vertices[faces]
    edge1 = fverts[:,1,:] - fverts[:,0,:]
    edge2 = fverts[:,2,:] - fverts[:,0,:]

    # Determinant definition of area
    area = 0.5 * torch.abs(edge1[:,0] * edge2[:,1]  - edge1[:,1] * edge2[:,0])

    # Debugging
    # print(area[0])
    # print(fverts[0])
    # exit()

    if return_fareas == True:
        return area
    else:
        return torch.sum(area)

# ==================== Loss Wrapper Class ===============================
### Naming conventions
### - "loss": will be plotted in the plot_uv function
### - "edge": loss will be plotted over edges, else triangles
from collections import defaultdict

class UVLoss:
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.currentloss = defaultdict(dict)
        self.device = device
        self.count = 0

        ### Sanity checks
        # Can only have one of seploss or gradloss
        # assert not (self.args.lossedgeseparation and self.args.lossgradientstitching), f"Can only have one type of edge loss!"

        # Record loss names (for visualization)
        self.lossnames = []

        if self.args.stitchingloss is not None:
            for lossname in self.args.stitchingloss:
                self.lossnames.append(lossname)

        if self.args.lossdistortion:
            self.lossnames.append("distortionloss")

    def clear(self):
        self.currentloss = defaultdict(dict)
        self.count = 0

    def computeloss(self, vertices = None, faces = None, uv = None, jacobians = None, initjacobs=None, seplossdelta=0.1,
                    weights = None, stitchweights = None, source = None, keepidxs = None, mesh = None,
                    predj = None, normalgrid = None):
        loss = torch.tensor([0.], device=self.device).double()
        # Ground truth
        if self.args.gtuvloss:
            if source is not None and source.fgroups is not None:
                gt_uvs = source.get_loaded_data('gt_uvs')
                gtuvloss = torch.zeros((len(gt_uvs), 2), device=gt_uvs.device).double()
                fgroups = source.fgroups

                for fgroup in np.unique(fgroups):
                    fgroupmask = np.where(fgroups == fgroup)
                    groupuvs = uv[fgroupmask].reshape(-1, 2)
                    groupgt = gt_uvs.reshape(-1, 3, 2)[fgroupmask].reshape(-1, 2)

                    # NOTE: gt uvs are centered so we center the predicted uvs as well
                    groupuvs = groupuvs - torch.mean(groupuvs, dim=0, keepdim=True).detach()

                    # Make sure gt uvs are centered
                    groupgt = groupgt - torch.mean(groupgt, dim=0, keepdim=True)

                    # Center and rotate predicted uvs by optimal rotation
                    if hasattr(self.args, 'optrot') and self.args.optrot:
                        with torch.no_grad():
                            M = groupuvs.T @ groupgt
                            # SVD of M to get optimal rotation (NO SCALING)
                            u, s, vt = torch.linalg.svd(M)
                            R = u @ vt

                            # NO FLIPS!!
                            baddet = torch.where(torch.det(R) < 0)[0]
                            if torch.det(R) <= 0:
                                vt[-1] *= -1
                                R = u @ vt
                                assert torch.det(R) > 0

                            groupgt = groupgt @ R.T

                    groupidxs = torch.arange(len(gt_uvs)).reshape(-1, 3)[fgroupmask].reshape(-1)
                    gtuvloss[groupidxs] = torch.nn.functional.mse_loss(groupuvs, groupgt, reduction='none')
            else:
                # NOTE: gt uvs are centered so we center the predicted uvs as well
                preduv = uv.reshape(-1, 2)
                preduv = preduv - torch.mean(preduv, dim=0, keepdim=True).detach()

                gt_uvs = source.get_loaded_data('gt_uvs')

                # Make sure gt uvs are centered
                gt_uvs = gt_uvs - torch.mean(gt_uvs, dim=0, keepdim=True)

                # Center and rotate predicted uvs by optimal rotation
                if hasattr(self.args, 'optrot') and self.args.optrot:
                    with torch.no_grad():
                        M = preduv.T @ gt_uvs
                        # SVD of M to get optimal rotation (NO SCALING)
                        u, s, vt = torch.linalg.svd(M)
                        R = u @ vt

                        # NO FLIPS!!
                        baddet = torch.where(torch.det(R) < 0)[0]
                        if torch.det(R) <= 0:
                            vt[-1] *= -1
                            R = u @ vt
                            assert torch.det(R) > 0

                        gt_uvs = gt_uvs @ R.T
                ## Debugging: show uvs pre and post rotation
                # if self.args.debug:
                #     import matplotlib.pyplot as plt
                #     from matplotlib.tri import Triangulation
                #     import os

                #     triangles = np.arange(len(gt_uvs)).reshape(-1, 3)
                #     fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                #     cmap = plt.get_cmap("tab20")

                #     tris = Triangulation(gt_uvs[:, 0].detach().cpu().numpy(), gt_uvs[:, 1].detach().cpu().numpy(), triangles=triangles)
                #     axs[0].set_title("GT")
                #     axs[0].tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                #                         linewidth=0.1, edgecolor="black")

                #     rotuv = gt_uvs @ R.T
                #     tris = Triangulation(rotuv[:, 0].detach().cpu().numpy(), rotuv[:, 1].detach().cpu().numpy(), triangles=triangles)
                #     axs[1].set_title("GT Rot")
                #     axs[1].tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                #                         linewidth=0.1, edgecolor="black")

                #     tris = Triangulation(preduv[:, 0].detach().cpu().numpy(), preduv[:, 1].detach().cpu().numpy(), triangles=triangles)
                #     axs[2].set_title("Pred UV")
                #     axs[2].tripcolor(tris, facecolors=np.ones(len(triangles)), cmap=cmap,
                #                         linewidth=0.1, edgecolor="black")

                #     for ax in axs:
                #         ax.axis('off')
                #         ax.axis('equal')
                #     plt.savefig(os.path.join(source.source_dir, "..", "..", "debugrot.png"))
                #     plt.close(fig)
                #     plt.cla()

                gtuvloss = torch.nn.functional.mse_loss(preduv, gt_uvs, reduction='none')

            loss += torch.mean(torch.sum(gtuvloss, axis=1))
            self.currentloss[self.count]['gtuvloss'] = np.sum(gtuvloss.detach().cpu().numpy(), axis=1)

        # Edge ground truth
        if self.args.gtedgeloss:
            gtweightloss = torch.nn.functional.mse_loss(weights, source.get_loaded_data('gt_weights'), reduction='none') # E
            loss += torch.mean(gtweightloss)
            self.currentloss[self.count]['gtweightloss'] = gtweightloss.detach().cpu().numpy()

        # Network ground truth
        if self.args.gtnetworkloss:
            gtjloss = torch.nn.functional.mse_loss(predj.squeeze()[:,:2,:], source.get_loaded_data('gt_jacobians'), reduction='none') # F x 3 x 2
            gtweightloss = torch.nn.functional.mse_loss(weights, source.get_loaded_data('gt_weights'), reduction='none') # E
            loss += torch.mean(torch.sum(gtjloss, axis=[1,2]))
            loss += torch.mean(gtweightloss)
            self.currentloss[self.count]['gtjloss'] = np.sum(np.sum(gtjloss.detach().cpu().numpy(), axis=1), axis=1)
            self.currentloss[self.count]['gtweightloss'] = gtweightloss.detach().cpu().numpy()

        # Just jacobian ground truth
        if self.args.gtjacobianloss:
            gtjloss = torch.nn.functional.mse_loss(predj.squeeze()[:,:2,:], source.get_loaded_data('gt_jacobians'), reduction='none') # F x 3 x 2
            loss += torch.mean(torch.sum(gtjloss, axis=[1,2]))
            self.currentloss[self.count]['gtjloss'] = np.sum(np.sum(gtjloss.detach().cpu().numpy(), axis=1), axis=1)

        # UV range loss (keep UVs in 0-1)
        if self.args.uvrangeloss:
            uvrange = torch.nn.functional.relu(uv - 1) + torch.nn.functional.relu(-uv)
            loss += self.args.uvrangeloss_weight * torch.sum(torch.sum(uvrange, axis=1))
            self.currentloss[self.count]['uvrangeloss'] = np.sum(uvrange.detach().cpu().numpy(), axis=1)

        # UV log loss (barrier fcn to keep UVs in 0-1)
        if self.args.uvlogloss:
            from source_njf.utils import dclamp
            uvlog = -torch.log(dclamp(uv, 0, 1) + 1e-10) - torch.log(1 - dclamp(uv, 0, 1) + 1e-10) + 2 * torch.log(torch.tensor(0.5, device=uv.device))
            loss += self.args.uvrangeloss_weight * torch.sum(torch.sum(uvlog, axis=1))
            self.currentloss[self.count]['uvlogloss'] = np.sum(uvlog.detach().cpu().numpy(), axis=1)

        # Normal grid loss (from NUVO)
        if self.args.normalloss:
            if normalgrid is None:
                raise ValueError(f"Computing normal loss requires the normal grid to be passed!")

            # First normalize UVs to be within (0-1)
            normuv = uv.clone()

            from source_njf.utils import normalize_uv
            with torch.no_grad():
                # Normalize to [-1, 1]
                normalize_uv(normuv)
                normuv = normuv * 2

            gt_fnormals = source.get_loaded_data('fnormals')

            # Bilinear interpolation of UVs to get the vertex samples from grid, then avg per face to get pred face normals
            pred_vnormals = torch.nn.functional.grid_sample(normalgrid.unsqueeze(0), normuv.unsqueeze(0), mode='bilinear')
            pred_fnormals = torch.mean(pred_vnormals, dim=-1).squeeze(0).transpose(1,0)

            normalloss = torch.nn.functional.mse_loss(pred_fnormals, gt_fnormals, reduction='none')
            loss += self.args.normalloss_weight * torch.mean(torch.sum(normalloss, axis=1))
            self.currentloss[self.count]['normalloss'] = self.args.normalloss_weight * np.sum(normalloss.detach().cpu().numpy(), axis=1)

        # Intersetion loss
        if self.args.intersectionloss:
            assert mesh is not None, f"Must provide mesh to compute intersection loss!"

            # Need to construct point/triangle arrays for 3 cases
            # (1) triangles with 3 neighbors (N = 3)
            # (2) triangles with 2 neighbors (N = 2)
            # (3) triangles with 1 neighbor (N = 1)
            pointsn3 = []
            trisn3 = []
            ordern3 = []

            pointsn2 = []
            trisn2 = []
            ordern2 = []

            pointsn1 = []
            trisn1 = []
            ordern1 = []

            for fi in range(len(mesh.faces)):
                # Get neighbors
                face = mesh.topology.faces[fi]
                neighborfis = [f.index for f in face.adjacentFaces()]

                if len(neighborfis) == 3:
                    pointsn3.append(uv[fi])
                    trisn3.append(uv[neighborfis])
                    ordern3.append(fi)
                elif len(neighborfis) == 2:
                    pointsn2.append(uv[fi])
                    trisn2.append(uv[neighborfis])
                    ordern2.append(fi)
                elif len(neighborfis) == 1:
                    pointsn1.append(uv[fi])
                    trisn1.append(uv[neighborfis])
                    ordern1.append(fi)

            # Convert to tensors
            ordered_isect = np.zeros(len(mesh.faces))
            isectloss = []
            if len(pointsn3) > 0:
                pointsn3 = torch.stack(pointsn3)
                trisn3 = torch.stack(trisn3)
                isectloss3 = intersectionloss(pointsn3, trisn3) # F1 x 3 x 3
                isectloss.append(torch.sum(isectloss3, dim=[1,2]))
                ordered_isect[ordern3] = isectloss[-1].detach().cpu().numpy()
            if len(pointsn2) > 0:
                pointsn2 = torch.stack(pointsn2)
                trisn2 = torch.stack(trisn2)
                isectloss2 = intersectionloss(pointsn2, trisn2) # F2 x 2 x 3
                isectloss.append(torch.sum(isectloss2, dim=[1,2]))
                ordered_isect[ordern2] = isectloss[-1].detach().cpu().numpy()
            if len(pointsn1) > 0:
                pointsn1 = torch.stack(pointsn1)
                trisn1 = torch.stack(trisn1)
                isectloss1 = intersectionloss(pointsn1, trisn1) # F3 x 1 x 3
                isectloss.append(torch.sum(isectloss1, dim=[1,2]))
                ordered_isect[ordern1] = isectloss[-1].detach().cpu().numpy()

            # TODO: track correspondence to tris and visualize somehow???
            isectloss = torch.cat(isectloss, dim=0) # F
            assert len(isectloss) == len(mesh.faces), f"Intersection loss length mismatch: {len(isectloss)} vs. faces: {len(mesh.faces)}"

            loss += self.args.intersectionloss_weight * torch.mean(isectloss)
            self.currentloss[self.count]['intersectionloss'] = ordered_isect

        # Autocuts
        # if self.args.lossautocut:
        #     acloss = autocuts(vertices, faces, jacobians, uv, self.args.seplossweight, seplossdelta)
        #     loss += acloss
        #     self.currentloss[self.count]['autocuts'] = acloss.detach().item()

        ## Stitching loss: can be vertex separation, edge separation, or gradient stitching
        ## Options: {l1/l2 distance}, {seamless transformation}, {weighting}
        if self.args.stitchingloss is not None:
            edgesep, stitchingdict, weightdict = stitchingloss(vertices, faces, uv.reshape(-1, 2), self.args.stitchingloss, self.args,
                                                      stitchweights=stitchweights, source = source,
                                                      keepidxs = keepidxs)
            for k, v in stitchingdict.items():
                self.currentloss[self.count][k] = weightdict[k] * v.detach().cpu().numpy()
                loss += weightdict[k] * torch.mean(v)

            # Edge sep always goes into lossdict for visualization purposes
            self.currentloss[self.count]['edgeseparation'] = edgesep.detach().cpu().numpy()

        ### Distortion loss: can be ARAP, ASAP, symmetric dirichlet
        distortionenergy = None
        if self.args.lossdistortion == "arap":
            distortionenergy = arap(vertices, faces, uv, paramtris=uv,
                                device=self.device, renormalize=False,
                                return_face_energy=True, timeit=False, elen_normalize=self.args.arapnorm)
            self.currentloss[self.count]['distortionloss'] = self.args.distortion_weight * distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += self.args.distortion_weight * torch.mean(distortionenergy)

        if self.args.lossdistortion == "asap":
            distortionenergy = asap(vertices, faces, uv, paramtris=uv, device=self.device,
                                return_face_energy=True, timeit=False, elen_normalize=self.args.arapnorm)
            self.currentloss[self.count]['distortionloss'] = self.args.distortion_weight * distortionenergy.detach().cpu().numpy()
            loss += self.args.distortion_weight * torch.sum(distortionenergy)

        if self.args.lossdistortion == "mips":
            distortionenergy = mips(vertices, faces, uv, paramtris=uv, device=self.device,
                                return_face_energy=True, timeit=False)
            self.currentloss[self.count]['distortionloss'] = self.args.distortion_weight * distortionenergy.detach().cpu().numpy()
            loss += self.args.distortion_weight * torch.sum(distortionenergy)

        if self.args.lossdistortion == "confvar":
            distortionenergy = confvar(vertices, faces, uv, paramtris=uv, device=self.device,
                                amips=self.args.amips, amips_weight=self.args.amips_weight, return_face_energy=True, timeit=False)
            self.currentloss[self.count]['distortionloss'] = self.args.distortion_weight * distortionenergy.detach().cpu().numpy()
            loss += self.args.distortion_weight * torch.sum(distortionenergy)

        if self.args.lossdistortion == "dirichlet":
            distortionenergy = symmetricdirichlet(vertices, faces, jacobians.squeeze())
            self.currentloss[self.count]['distortionloss'] = self.args.distortion_weight * distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += self.args.distortion_weight * torch.mean(distortionenergy)

        # Cut sparsity loss
        if self.args.sparsecutsloss:
            sparseloss = parabolaloss(weights).mean()
            self.currentloss[self.count]['sparsecutsloss'] = self.args.sparsecuts_weight * sparseloss.detach().cpu().numpy()
            loss += self.args.sparsecuts_weight * sparseloss

        self.currentloss[self.count]['total'] = loss.item()
        self.count += 1

        return loss

    def exportloss(self):
        return self.currentloss

# ==================== Edge Cut Sparsity Losses ===============================
def parabolaloss(weights):
    """ Penalizes weights at 0.5 and none at 0/1. Min 0 and max 1 assuming weights are in [0,1]. """
    return -4 * (weights - 0.5) ** 2 + 1

# ==================== Distortion Energies ===============================
# TODO: batch this across multiple meshes
# NOTE: if given an initialization jacobian, then predicted jacob must be 2x2 matrix
def symmetricdirichlet(vs, fs, jacob=None, init_jacob=None):
    # Jacob: F x 2 x 3
    # TODO: Below can be precomputed
    # Get face areas
    from meshing import Mesh
    from meshing.analysis import computeFaceAreas
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    computeFaceAreas(mesh)
    fareas = torch.from_numpy(mesh.fareas).to(vs.device)

    if jacob is not None:
        mdim = 4 # NOTE: We will ALWAYS assume a 2x2 transformation so inverse is well defined
        if init_jacob is not None:
            # NOTE: assumes left multiplication => J' x J_I x Fvert
            # Use 2x2 upper diagonal of predicted jacobian
            if init_jacob.shape[1] > 2:
                init_jacob = init_jacob[:,:2,:]

            jacob2 = torch.matmul(jacob[:,:2,:2], init_jacob) # F x 2 x 3
            # jacob2 = torch.matmul(init_jacob, jacob[:,:2,:2]) # F x 2 x 2

            # Need final jacobian to be 2x2
            if jacob2.shape[2] > 2:
                jacob2 = torch.matmul(jacob2, jacob2.transpose(1,2))
        # NOTE: This assumes jacob matrix is size B x J1 x J2
        elif jacob.shape[2] > 2:
            # Map jacob to 2x2 by multiplying against transpose
            jacob2 = torch.matmul(jacob, jacob.transpose(1,2))
        else:
            jacob2 = jacob
        try:
            invjacob = torch.linalg.inv(jacob2)
        except Exception as e:
            print(f"Torch inv error on jacob2: {e}")
            invjacob = torch.linalg.pinv(jacob2)
        energy = fareas * (torch.sum((jacob2 * jacob2).reshape(-1, mdim), dim=-1) + torch.sum((invjacob * invjacob).reshape(-1, mdim), dim=-1) - 4)
    else:
        # Rederive jacobians from UV values
        raise NotImplementedError("Symmetric dirichlet with manual jacobian calculation not implemented yet!")
    return energy

def arap(vertices, faces, param, return_face_energy=True, paramtris=None, renormalize=False,
         elen_normalize=False, fixcot=False,
         face_weights=None, normalize_filter=0, device=torch.device("cpu"), verbose=False, timeit=False, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: sometimes denominator will be 0 i.e. area of triangle is 0 -> cotangent in this case is infty, default to 1e5
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3])

    if fixcot:
        cot_full = torch.abs(cot_full)

    D = torch.stack([torch.diag(cot_full[:,i]) for i in range(cot_full.shape[1])])
    e_full = torch.stack([e1, e2, e3]) # 3 x F x 2
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    ## Procrustes: SVD of P'P^T where P, P' are point sets for source and target, respectively
    crosscov = torch.matmul(torch.matmul(e_p_full.permute(1, 2, 0), D), e_full.permute(1, 0, 2),)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    # NOTE: This is U^T
    U = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    # NOTE: This is V
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    R = torch.matmul(Ut.transpose(2,1), V.transpose(2,1)).to(device) # F x 2 x 2

    ## NOTE: Sanity check the SVD
    S = torch.stack([torch.diag(torch.tensor([S1[i], S2[i]])) for i in range(len(S1))]).to(S1.device)
    checkcov = U.transpose(2,1) @ S @ V.transpose(2,1)
    torch.testing.assert_close(crosscov.reshape(-1, 2, 2), checkcov)

    # Sometimes rotation is opposite orientation: just check with determinant and flip
    # NOTE: Can flip sign of det by flipping sign of last column of V
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        print(f"ARAP warning: found {len(baddet)} flipped rotations.")
        V[baddet, :, 1] *= -1
        R = torch.matmul(Ut.transpose(2,1), V.transpose(2,1)).to(device) # F x 2 x 2
        assert torch.all(torch.det(R) >= 0)

    edge_tmp = torch.stack([e1, e2, e3], dim=2)
    rot_edges = torch.matmul(R, edge_tmp) # F x 2 x 3
    rot_e_full = rot_edges.permute(2, 0, 1) # 3 x F x 2
    cot_full = cot_full.reshape(cot_full.shape[0], cot_full.shape[1]) # 3 x F

    if renormalize == True:
        # ARAP-minimizing scaling of parameterization edge lengths
        if face_weights is not None:
            keepfs = torch.where(face_weights > normalize_filter)[0]
        else:
            keepfs = torch.arange(rot_e_full.shape[1])

        num = torch.sum(cot_full[:,keepfs] * torch.sum(rot_e_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))
        denom = torch.sum(cot_full[:,keepfs] * torch.sum(e_p_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))

        ratio = max(num / denom, 1e-5)
        if verbose == True:
            print(f"Scaling param. edges by ARAP-minimizing scalar: {ratio}")

        e_p_full *= ratio

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e_p_full)) or not torch.all(torch.isfinite(rot_e_full)):
        print(f"ARAP: non-finite elements found")
        return None

    # Compute face-level distortions
    # from meshing import Mesh
    # from meshing.analysis import computeFaceAreas
    # mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
    # computeFaceAreas(mesh)
    # fareas = torch.from_numpy(mesh.fareas).to(vertices.device)

    # NOTE: We normalize by mean edge length b/w p and e b/c for shrinking ARAP is bounded by edge length
    # Normalizing by avg edge length b/w p and e => ARAP bounded by 2 on BOTH SIDES
    if elen_normalize:
        mean_elen = (torch.linalg.norm(e_full, dim=2) + torch.linalg.norm(e_p_full, dim=2))/2
        arap_tris = torch.sum(cot_full * 1/mean_elen * torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2, dim=0) # F x 1
    else:
        arap_tris = torch.sum(cot_full * torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2, dim=0) # F x 1

    if timeit == True:
        print(f"ARAP calculation: {time.time()-t0:0.5f}")

    # Debugging: show rotated edges along with parameterization
    # import polyscope as ps
    # ps.init()
    # param_f1 = param[0]
    # # Normalize the param so first vertex is at 0,0
    # param_f1 = param_f1 - param_f1[0]
    # og_f1 = local_tris[0] # 3 x 2
    # rot_f1 = R[0]
    # new_f1 = torch.matmul(rot_f1, og_f1.transpose(1,0)).transpose(1,0)
    # print(new_f1)
    # og_curve = ps.register_curve_network("og triangle", og_f1.detach().cpu().numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,1,0])
    # param_curve = ps.register_curve_network("UV", param_f1.detach().cpu().numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,0,1])
    # rot_curve = ps.register_curve_network("rot triangle", new_f1.detach().cpu().numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[1,0,0])
    # ps.show()

    # # Compute energies
    # print(e_p_full.shape)
    # print(e_full.shape)
    # print(arap_tris[0])
    # print(torch.sum(cot_full[:,0] * torch.linalg.norm(e_p_full[:,0,:] - e_full[:,0,:], dim=1) ** 2))

    # raise

    if return_face_energy == False:
        return torch.mean(arap_tris)

    return arap_tris


### NOTE: Follows setup for ASAP, but loss is S1/S2 + S2/S1
# MIPS energy: conformal + penalizes shrinking
def mips(vertices, faces, param, return_face_energy=True, paramtris=None,
         elen_normalize=False, device=torch.device("cpu"), verbose=False, timeit=False,
         fixcot=False, fareas=None, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: This only gives cotangent weight for SOUP (one side of edge)
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3])

    if fixcot:
        cot_full = torch.abs(cot_full)

    D = torch.stack([torch.diag(cot_full[:,i]) for i in range(cot_full.shape[1])])
    e_full = torch.stack([e1, e2, e3]) # 3 x F x 2
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    ## Procrustes: SVD of P'P^T where P, P' are point sets for source and target, respectively
    crosscov = torch.matmul(torch.matmul(e_p_full.permute(1, 2, 0), D), e_full.permute(1, 0, 2),)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R

    # NOTE: We take the squares to avoid issues with negative singular values (when detR is negative)
    conformal_tris = S1**2/S2**2 + S2**2/S1**2 - 2

    if return_face_energy == False:
        return torch.mean(conformal_tris)

    return conformal_tris

## Conformal variance: aims to minimize the variance across all the singular values in the mesh
def confvar(vertices, faces, param, amips=False, amips_weight=0.2, return_face_energy=True, paramtris=None,
         elen_normalize=False, device=torch.device("cpu"), verbose=False, timeit=False,
         fixcot=False, fareas=None, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: This only gives cotangent weight for SOUP (one side of edge)
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3])

    if fixcot:
        cot_full = torch.abs(cot_full)

    D = torch.stack([torch.diag(cot_full[:,i]) for i in range(cot_full.shape[1])])
    e_full = torch.stack([e1, e2, e3]) # 3 x F x 2
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    ## Procrustes: SVD of P'P^T where P, P' are point sets for source and target, respectively
    crosscov = torch.matmul(torch.matmul(e_p_full.permute(1, 2, 0), D), e_full.permute(1, 0, 2),)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R

    S = torch.stack([S1, S2], dim=1)
    Smean = torch.mean(S)
    Svar = torch.sum((S - Smean) ** 2, dim=1)

    face_distortion = Svar
    if amips:
        amips_d = S1**2/S2**2 + S2**2/S1**2 - 2
        face_distortion = Svar + amips_weight * amips_d

    if return_face_energy == False:
        return torch.mean(face_distortion)

    return face_distortion

def asap(vertices, faces, param, return_face_energy=True, paramtris=None,
         elen_normalize=False, device=torch.device("cpu"), verbose=False, timeit=False,
         fixcot=False, fareas=None, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: This only gives cotangent weight for SOUP (one side of edge)
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3])

    if fixcot:
        cot_full = torch.abs(cot_full)

    D = torch.stack([torch.diag(cot_full[:,i]) for i in range(cot_full.shape[1])])
    e_full = torch.stack([e1, e2, e3]) # 3 x F x 2
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    ## Procrustes: SVD of P'P^T where P, P' are point sets for source and target, respectively
    crosscov = torch.matmul(torch.matmul(e_p_full.permute(1, 2, 0), D), e_full.permute(1, 0, 2),)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    ## Manual 2x2 SVD
    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    # NOTE: This is U^T
    Ut = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    # NOTE: This is V
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    ## ASAP: USV where S is average of the two singular values
    Smean = (S1 + S2) / 2
    S = torch.stack([torch.diag(torch.tensor([Smean[i], Smean[i]])) for i in range(len(Smean))]).to(Smean.device)
    R = torch.matmul(torch.matmul(Ut.transpose(2,1), S), V.transpose(2,1)).to(device) # F x 2 x 2

    ## NOTE: Sanity check the SVD
    S = torch.stack([torch.diag(torch.tensor([S1[i], S2[i]])) for i in range(len(S1))]).to(S1.device)
    checkcov = Ut.transpose(2,1) @ S @ V.transpose(2,1)
    torch.testing.assert_close(crosscov.reshape(-1, 2, 2), checkcov)

    # Sometimes rotation is opposite orientation: just check with determinant and flip
    # NOTE: Can flip sign of det by flipping sign of last column of V
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        print(f"ASAP warning: found {len(baddet)} flipped rotations.")
        V[baddet, :, 1] *= -1
        R = torch.matmul(torch.matmul(Ut.transpose(2,1), S), V.transpose(2,1)).to(device) # F x 2 x 2
        assert torch.all(torch.det(R) >= 0)

    edge_tmp = torch.stack([e1, e2, e3], dim=2)
    rot_edges = torch.matmul(R, edge_tmp) # F x 2 x 3
    rot_e_full = rot_edges.permute(2, 0, 1) # 3 x F x 2
    cot_full = cot_full.reshape(cot_full.shape[0], cot_full.shape[1]) # 3 x F

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e_p_full)) or not torch.all(torch.isfinite(rot_e_full)):
        print(f"ASAP: non-finite elements found")
        return None

    # NOTE: We normalize by mean edge length b/w p and e b/c for shrinking ARAP is bounded by edge length
    # Normalizing by avg edge length b/w p and e => ARAP bounded by 2 on BOTH SIDES
    if elen_normalize:
        mean_elen = (torch.linalg.norm(e_full, dim=2) + torch.linalg.norm(e_p_full, dim=2))/2
        asap_tris = torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2
        if fareas is not None:
            asap_tris = torch.sum(fareas * 1/mean_elen * asap_tris, dim=0) # F x 1
        else:
            asap_tris = torch.sum(1/mean_elen * asap_tris, dim=0) # F x 1
    else:
        asap_tris = torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2
        if fareas is not None:
            asap_tris = torch.sum(fareas * asap_tris, dim=0) # F x 1
        else:
            asap_tris = torch.sum(asap_tris, dim=0) # F x 1

    if timeit == True:
        print(f"ASAP calculation: {time.time()-t0:0.5f}")

    if return_face_energy == False:
        return torch.mean(asap_tris)

    return asap_tris

def asap_old(vertices, faces, param, return_face_energy=True, paramtris=None,
         elen_normalize=False, device=torch.device("cpu"), verbose=False, timeit=False,
         fixcot=False, fareas=None, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: This only gives cotangent weight for SOUP (one side of edge)
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3])

    if fixcot:
        cot_full = torch.abs(cot_full)

    D = torch.stack([torch.diag(cot_full[:,i]) for i in range(cot_full.shape[1])])
    e_full = torch.stack([e1, e2, e3]) # 3 x F x 2
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    ## Procrustes: SVD of P'P^T where P, P' are point sets for source and target, respectively
    crosscov = torch.matmul(torch.matmul(e_full.permute(1, 2, 0), D), e_p_full.permute(1, 0, 2),)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    ## Manual 2x2 SVD
    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    # NOTE: This is U^T
    Ut = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    # NOTE: This is V
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    ## ASAP: USV where S is average of the two singular values
    Smean = (S1 + S2) / 2
    S = torch.stack([torch.diag(torch.tensor([Smean[i], Smean[i]])) for i in range(len(Smean))]).to(Smean.device)
    R = torch.matmul(torch.matmul(V, S), Ut).to(device) # F x 2 x 2

    ## NOTE: Sanity check the SVD
    S = torch.stack([torch.diag(torch.tensor([S1[i], S2[i]])) for i in range(len(S1))]).to(S1.device)
    checkcov = Ut.transpose(2,1) @ S @ V.transpose(2,1)
    torch.testing.assert_close(crosscov.reshape(-1, 2, 2), checkcov)

    # Sometimes rotation is opposite orientation: just check with determinant and flip
    # NOTE: Can flip sign of det by flipping sign of last column of V
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        print(f"ASAP warning: found {len(baddet)} flipped rotations.")
        V[baddet, :, 1] *= -1
        R = torch.matmul(torch.matmul(V, S), Ut).to(device) # F x 2 x 2
        assert torch.all(torch.det(R) >= 0)

    edge_tmp = torch.stack([e1, e2, e3], dim=2)
    rot_edges = torch.matmul(R, edge_tmp) # F x 2 x 3
    rot_e_full = rot_edges.permute(2, 0, 1) # 3 x F x 2
    cot_full = cot_full.reshape(cot_full.shape[0], cot_full.shape[1]) # 3 x F

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e_p_full)) or not torch.all(torch.isfinite(rot_e_full)):
        print(f"ASAP: non-finite elements found")
        return None

    # NOTE: We normalize by mean edge length b/w p and e b/c for shrinking ARAP is bounded by edge length
    # Normalizing by avg edge length b/w p and e => ARAP bounded by 2 on BOTH SIDES
    if elen_normalize:
        mean_elen = (torch.linalg.norm(e_full, dim=2) + torch.linalg.norm(e_p_full, dim=2))/2
        asap_tris = torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2
        if fareas is not None:
            asap_tris = torch.sum(fareas * 1/mean_elen * asap_tris, dim=0) # F x 1
        else:
            asap_tris = torch.sum(1/mean_elen * asap_tris, dim=0) # F x 1
    else:
        asap_tris = torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2
        if fareas is not None:
            asap_tris = torch.sum(fareas * asap_tris, dim=0) # F x 1
        else:
            asap_tris = torch.sum(asap_tris, dim=0) # F x 1

    if timeit == True:
        print(f"ASAP calculation: {time.time()-t0:0.5f}")

    if return_face_energy == False:
        return torch.mean(asap_tris)

    return asap_tris

# Edge distortion: MSE between UV edge norm and original edge norm
def edgedistortion(vs, fs, uv):
    """vs: V x 3
       fs: F x 3
       uv: F x 3 x 2 """
    from meshing import Mesh
    from meshing.analysis import computeFacetoEdges

    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    computeFacetoEdges(mesh)

    # Edge lengths per triangle
    f_elens = []
    uv_elens = []
    fverts = vs[fs] # F x 3 x 3
    for fi in range(len(fverts)):
        f_elens.append([torch.linalg.norm(fverts[fi, 1] - fverts[fi, 0]), torch.linalg.norm(fverts[fi, 2] - fverts[fi, 1]), torch.linalg.norm(fverts[fi, 2] - fverts[fi, 0])])
        uv_elens.append([torch.linalg.norm(uv[fi, 1] - uv[fi, 0]), torch.linalg.norm(uv[fi, 2] - uv[fi, 1]), torch.linalg.norm(uv[fi, 2] - uv[fi, 0])])

    f_elens = torch.tensor(f_elens, device=uv.device)
    uv_elens = torch.tensor(uv_elens, device=uv.device)
    energy = torch.nn.functional.mse_loss(uv_elens, f_elens, reduction='none')

    # Sum over triangles
    energy = torch.sum(energy, dim=1)

    return energy

## ===== Triangle Intersection Checks (Pytorch) =====
# check that all points of the other triangle are on the same side of the triangle after mapping to barycentric coordinates.
# returns true if all points are outside on the same side
def intersectionloss(points, checktri):
    """
        points: B x 3 x 2 points to check
        checktri: B x N x 3 x 2 triangles to check against (each set in points checks against N triangles)
    """
    points = points.unsqueeze(1)
    d2 = points - checktri[:, :, [2], :] # B x N x 3 x 2
    dX21 = checktri[:, :, 2, 0] - checktri[:, :, 1, 0] # B x N
    dY12 = checktri[:, :, 1, 1] - checktri[:, :, 2, 1] # B x N
    D = dY12 * (checktri[:, :, 0, 0] - checktri[:, :, 2, 0]) + dX21 * (checktri[:, :, 0, 1] - checktri[:, :, 2, 1]) # B x N
    S = dY12.unsqueeze(-1) * d2[:, :, :, 0] + dX21.unsqueeze(-1) * d2[:, :, :, 1] # B x N x 3
    T = (checktri[:, :, 2, 1] - checktri[:, :, 0, 1]).unsqueeze(-1) * d2[:, :, :, 0] + (checktri[:, :, 0, 0] - checktri[:, :, 2, 0]).unsqueeze(-1) * d2[:, :, :, 1] # B x N x 3

    Dsign = torch.sign(D).unsqueeze(-1)
    Scheck = Dsign * S
    Tcheck = Dsign * T
    STcheck = Dsign * (S + T)

    # In loss form: we can sum over the violation of the inequalities
    # Standard check: Scheck <= 0 or Tcheck <= 0 or STcheck >= D => all points are outside
    # Loss form: mask * (torch.max(0, Scheck) + torch.max(0, Tcheck) + torch.max(0, D - STcheck))
    #   where mask is 1 if all equalities are violated (so intersection is true)
    intersections = ~(torch.all(Scheck <= torch.zeros_like(Scheck), dim= 2) | torch.all(Tcheck <= torch.zeros_like(Tcheck), dim= 2) | torch.all(STcheck >= D.unsqueeze(-1), dim=2)).unsqueeze(-1) # B x N
    intersectionloss = intersections * (torch.max(torch.zeros_like(Scheck), Scheck) + torch.max(torch.zeros_like(Tcheck), Tcheck) + torch.max(torch.zeros_like(STcheck), D.unsqueeze(-1) - STcheck)) # B x N x 3

    return intersectionloss

# Minimal implementation (without batching)
def sameside(points, checktri):
    """
        points: 3 x 2 points to check
        checktri: 3 x 2 triangle to check against
    """
    d2 = points - checktri[[2], :] # 3 x 2
    dX21 = checktri[2, 0] - checktri[1, 0]
    dY12 = checktri[1, 1] - checktri[2, 1]
    D = dY12 * (checktri[0, 0] - checktri[2, 0]) + dX21 * (checktri[0, 1] - checktri[2, 1])
    S = dY12 * d2[:, 0] + dX21 * d2[:, 1] # 3
    T = (checktri[2, 1] - checktri[0, 1]) * d2[:, 0] + (checktri[0, 0] - checktri[2, 0]) * d2[:, 1]

    if D < 0:
        sameside = torch.all(S >= 0) | torch.all(T >= 0) | torch.all(S + T <= D)
    else:
        sameside = torch.all(S <= 0) | torch.all(T <= 0) | torch.all(S + T >= D)

    return sameside

# ==================== Stitching Energies ===============================
def stitchingloss(vs, fs, uv, losstypes, args, stitchweights=None, source = None, keepidxs = None):
    from source_njf.utils import vertex_soup_correspondences
    from itertools import combinations

    uvpairs = uv[source.edge_vpairs.to(uv.device)] # E x 2 x 2 x 2
    elens = source.elens_nobound.to(uv.device)

    if keepidxs is not None:
        uvpairs = uvpairs[keepidxs]
        elens = elens[keepidxs]


    ## Edge separation loss
    if args.stitchdist == 'l2':
        edgesep = torch.sqrt(torch.sum(torch.nn.functional.mse_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2))
        edgesep = torch.mean(edgesep, dim=1) # E x 1
    elif args.stitchdist == 'l1':
        edgesep = torch.sum(torch.nn.functional.l1_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2)
        edgesep = torch.mean(edgesep, dim=1) # E x 1
    elif args.stitchdist == 'centroid':
        centroids = (uvpairs[:,0,:,:] + uvpairs[:,1,:,:])/2 # E x (e1, e2) x 2
        edgesep = torch.linalg.norm(centroids[:,0] - centroids[:,1], dim=1) # E x 1

    # Weight with edge lengths
    wedgesep = edgesep * elens

    # if stitchweights is None:
    #     stitchweights = torch.ones(len(vertexsep), device=uv.device)

    lossdict = {}
    weightdict = {}
    # NOTE: We assume everything can just use edges going forward ... uncomment if not true
    for losstype in losstypes:
        # if losstype == "vertexseploss": # Pairs x 2
        #     if args.seamlessvertexsep:
        #         lossdict[losstype] = stitchweights * torch.sum(seamlessvertexsep, dim=1)
        #     else:
        #         lossdict[losstype] = stitchweights * torch.sum(vertexsep, dim=1)
        #         weightdict[losstype] = args.vertexsep_weight

        if losstype == "edgecutloss": # E x 1
            if args.seamlessedgecut:
                edgecutloss = wedgesep * wedgesep/(wedgesep * wedgesep + args.seamlessdelta)
            else:
                edgecutloss = wedgesep
            lossdict[losstype] = edgecutloss
            weightdict[losstype] = args.edgecut_weight

        elif losstype == "weightededgecutloss":
            edgecutweights = source.get_loaded_data('edgecutweights')
            if args.seamlessedgecut:
                edgecutloss = edgecutweights * wedgesep * wedgesep/(wedgesep * wedgesep + args.seamlessdelta)
            else:
                edgecutloss = edgecutweights * wedgesep
            lossdict[losstype] = edgecutloss
            weightdict[losstype] = args.edgecut_weight

    return edgesep, lossdict, weightdict

# Compute loss based on vertex-vertex distances
def vertexseparation(vs, fs, uv, loss='l1'):
    """ uv: F * 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """
    from source_njf.utils import vertex_soup_correspondences
    from itertools import combinations
    from meshing import Mesh
    vcorrespondences = vertex_soup_correspondences(fs.detach().cpu().numpy())
    vpairs = []
    for ogv, vlist in sorted(vcorrespondences.items()):
        vpairs.extend(list(combinations(vlist, 2)))
    vpairs = torch.tensor(vpairs, device=uv.device)
    uvpairs = uv[vpairs] # V x 2 x 2

    if loss == "l1":
        separation = torch.nn.functional.l1_loss(uvpairs[:,0], uvpairs[:,1], reduction='none')
    elif loss == 'l2':
        separation = torch.nn.functional.mse_loss(uvpairs[:,0], uvpairs[:,1], reduction='none')

    return separation

def uvgradloss(vs, fs, uv, return_edge_correspondence=False, loss='l2'):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """

    from source_njf.utils import edge_soup_correspondences
    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    # mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    # fconn, vconn = mesh.topology.export_edge_face_connectivity(mesh.faces)
    # fconn = np.array(fconn, dtype=int) # E x {f0, f1}
    # vconn = np.array(vconn, dtype=int) # E x {v0,v0'} x {v1, v1'}

    uvsoup = uv.reshape(-1, 2)
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(fs.detach().cpu().numpy())
    e1 = []
    e2 = []
    elens = []
    for k, v in sorted(edgecorrespondences.items()):
        # If only one correspondence, then it is a boundary
        if len(v) == 1:
            continue
        e1.append(uvsoup[list(v[0])])
        e2.append(uvsoup[list(v[1])])
        elens.append(np.linalg.norm(vs[k[0]] - vs[k[1]]))

    ef0 = torch.cat(e1) # E*2 x 2
    ef1 = torch.cat(e2) # E*2 x 2
    elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1)

    # Debugging: visualize the edge vectors
    # import polyscope as ps
    # ps.init()
    # ps_uv = ps.register_surface_mesh("uv", uvsoup, np.arange(len(uvsoup)).reshape(-1, 3), edge_width=1)

    # # Map vconn to flattened triangle indices
    # # NOTE: FCONN AND VCONN NO GOOD REDO
    # vertcurve = np.arange(len(ef0)).reshape(-1, 2)
    # ecolors = np.arange(len(vertcurve))

    # ps_curve = ps.register_curve_network("edgeside1", ef0, vertcurve, enabled=True)
    # ps_curve.add_scalar_quantity("ecolors", ecolors, defined_on='edges', enabled=True)
    # ps_curve = ps.register_curve_network("edgeside2", ef1, vertcurve, enabled=True)
    # ps_curve.add_scalar_quantity("ecolors", ecolors, defined_on='edges', enabled=True)
    # ps.show()

    # Compare the edge vectors (just need to make sure the edge origin vertices are consistent)
    e0 = ef0[::2] - ef0[1::2] # E x 2
    e1 = ef1[::2] - ef1[1::2] # E x 2

    # Weight each loss by length of 3D edge
    # from meshing.mesh import Mesh
    # mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    # elens = []
    # for i, edge in sorted(mesh.topology.edges.items()):
    #     if edge.onBoundary():
    #         continue
    #     elens.append(mesh.length(edge))
    # elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1)
    # elens /= torch.max(elens)

    if loss == "l1":
        separation = elens * torch.nn.functional.l1_loss(e0, e1, reduction='none')
    elif loss == 'l2':
        separation = elens * torch.nn.functional.mse_loss(e0, e1, reduction='none')
    elif loss == 'cosine':
        # Cosine similarity
        separation = elens.squeeze() * (1 - torch.nn.functional.cosine_similarity(e0, e1, eps=1e-8))

    if return_edge_correspondence:
        return separation, edgecorrespondences

    return separation

def uvseparation(vs, fs, uv, loss='l1'):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """
    from source_njf.utils import edge_soup_correspondences
    from meshing import Mesh
    uvsoup = uv.reshape(-1, 2)
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(fs.detach().cpu().numpy())
    e1 = []
    e2 = []
    for k, v in sorted(edgecorrespondences.items()):
        # If only one correspondence, then it is a boundary
        if len(v) == 1:
            continue
        e1.append(uvsoup[list(v[0])])
        e2.append(uvsoup[list(v[1])])

    ef0 = torch.stack(e1) # E*2 x 2
    ef1 = torch.stack(e2) # E*2 x 2

    # Weight each loss by length of 3D edge
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    elens = []
    for i, edge in sorted(mesh.topology.edges.items()):
        if edge.onBoundary():
            continue
        elens.append(mesh.length(edge))
    elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1,1)

    if loss == "l1":
        separation = elens * torch.nn.functional.l1_loss(ef0, ef1, reduction='none')
    elif loss == 'l2':
        separation = elens * torch.nn.functional.mse_loss(ef0, ef1, reduction='none')

    return separation, edgecorrespondences

def splitgradloss(vs, fs, uv, cosine_weight=1, mag_weight=1):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """

    from meshing import Mesh
    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    fconn, vconn = mesh.topology.export_edge_face_connectivity(mesh.faces)
    fconn = np.array(fconn, dtype=int) # E x {f0, f1}
    vconn = np.array(vconn, dtype=int) # E x {v0,v1}
    ef0 = uv[fconn[:,[0]], vconn[:,0]] # E x 2 x 2
    ef1 = uv[fconn[:,[1]], vconn[:,1]] # E x 2 x 2

    # Compare the edge vectors (just need to make sure the edge origin vertices are consistent)
    e0 = ef0[:,1] - ef0[:,0]
    e1 = ef1[:,1] - ef1[:,0]

    # Weight each loss by length of 3D edge
    elens = []
    for i, edge in sorted(mesh.topology.edges.items()):
        if edge.onBoundary():
            continue
        elens.append(mesh.length(edge))
    elens = torch.tensor(elens, device=uv.device)

    cosine_loss = -torch.nn.functional.cosine_similarity(e0, e1)
    mag_loss = torch.nn.functional.mse_loss(torch.norm(e0, dim=1)/elens, torch.norm(e1, dim=1)/elens, reduction='none')

    return elens * (cosine_weight * cosine_loss + mag_weight * mag_loss)

# Autocuts energy involves weighted sum of two measures
#   - Triangle distortion: symmetric dirichlet (sum of Fnorm of jacobian and inverse jacobian)
#   - Edge separation: f(L1 norm between UVs of corresponding vertices) w/ f = x^2/(x^2 + delta) (delta smoothing parameter -> converge to 0 over time)
def autocuts(vs, fs, js, uv, sepweight=1, delta=0.01, init_j=None):
    """vs: V x 3
       fs: F x 3
       js: F x 3 x 2
       uv: F x 3 x 2 """
    dirichlet = torch.mean(symmetricdirichlet(vs, fs, uv, js, init_j))

    # Get vertex correspondences per edge
    # NOTE: ordering of vertices in UV MUST BE SAME as ordering of vertices in fs
    separation = uvseparation(vs, fs, uv)
    separation = torch.mean((separation * separation)/(separation * separation + delta))
    energy = dirichlet + sepweight * separation
    return energy
