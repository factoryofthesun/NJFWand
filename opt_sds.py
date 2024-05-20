### SDS Optimization: Soft Poisson
from meshing.io import PolygonSoup
from meshing.mesh import Mesh
import numpy as np
import os
import igl
from scipy.sparse import diags
import torch
import argparse
from pathlib import Path
import dill as pickle
from collections import defaultdict
import torch
from PIL import Image
import torchvision
from torchvision.transforms import Resize
from source_njf.utils import clear_directory

parser = argparse.ArgumentParser()

parser.add_argument("--debug",
                    action='store_true', default=False)
parser.add_argument("--objdir",
                    help='path to obj file to optimize for',
                    type = str, default='./data/tetrahedron.obj')
parser.add_argument("--savedir",
                    help='path to save directory',
                    type = str, default='./outputs/optimization')
parser.add_argument("--savename",
                    help='name of experiment',
                    type = str, default=None)
parser.add_argument("--niters",
                    type = int, default=20000)
parser.add_argument("--init", type=str, choices={'slim', 'isometric', 'tutte'},
                    default='slim')
parser.add_argument("--no_weight_optimization", action='store_true')
parser.add_argument("--no_jacobian_optimization", action='store_true')

## LOSSES
parser.add_argument("--vertexseptype", type=str, choices={'l2', 'l1'}, default='l2')
parser.add_argument("--edgecutloss", action='store_true')
parser.add_argument("--edgecutloss_weight", type=float, default=0.1)

parser.add_argument("--distortionloss", action='store_true', default=False)
parser.add_argument("--distortionloss_weight", type=float, default=1)
parser.add_argument("--seamlessedgecut",
                    action='store_true', default=False)
parser.add_argument("--seamlessdelta", type=float, default=0.05)
parser.add_argument("--renderresolution", type=int, help="render resolution", default=512)
parser.add_argument("--resolution", type=int, help="supervision resolution", default=128)

parser.add_argument("--sdsloss", type=str, choices={'text2img', 'img2img', 'cascaded'}, default=None)
parser.add_argument("--imageloss", type=str, help="path to gt image texture if given", default=None)
parser.add_argument("--imagelossn", type=int, help="number of gt views for image loss", default=0)
parser.add_argument("--stage2weight", type=float, default=0.5)

parser.add_argument("--textureimg", type=str, default=None)
parser.add_argument('--texturetext', nargs="+", type=str, help="text description", default=None)
parser.add_argument("--interpmode", type=str, choices={'nearest', 'bilinear', 'bicubic'}, default='bilinear')

parser.add_argument("--edgeweightonly", help='optimize only edge weights. other valid pairs default to 0',
                    action='store_true', default=False)

parser.add_argument("--viziter", type=int, default=100)
parser.add_argument("--lr", type=float, default=3e-2)
parser.add_argument("--seplossdelta", type=float, default=0.1)
parser.add_argument("--cuteps", type=float, default=0.01)
parser.add_argument("--continuetrain",  action='store_true')

args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


savedir = args.savedir

### Weighting experiments
datadir = args.objdir
soup = PolygonSoup.from_obj(datadir)
mesh = Mesh(soup.vertices, soup.indices)
mesh.normalize()

ogvertices = torch.from_numpy(mesh.vertices).double().to(device)
ogfaces = torch.from_numpy(mesh.faces).long().to(device)

ogfverts = mesh.vertices[mesh.faces]
soupvs = torch.from_numpy(ogfverts.reshape(-1, 3)).double().to(device)
soupfs = torch.arange(len(soupvs)).reshape(-1, 3).long()

# Initialize the UV/Jacobians
from source_njf.utils import tutte_embedding, get_jacobian_torch, SLIM, get_local_tris, make_cut
vertices = torch.from_numpy(mesh.vertices).double().to(device)
faces = torch.from_numpy(mesh.faces).long().to(device)
fverts = vertices[faces]

ignoreset = []
if args.init == "slim":
    ## Enforce disk topology
    if len(mesh.topology.boundaries) == 0:
        from numpy.random import default_rng
        from source_njf.utils import generate_random_cuts
        rng = default_rng()
        n_cuts = rng.integers(1, 2)
        cutvs = generate_random_cuts(mesh, enforce_disk_topo=True, max_cuts = n_cuts)
    inituv = torch.from_numpy(SLIM(mesh)[0]).double().to(device)
    inituv = inituv[faces].reshape(-1, 2)
elif args.init == "isometric":
    vertices = fverts.reshape(-1, 3).double()
    faces = torch.arange(len(vertices)).reshape(-1, 3).long().to(device)
    inituv = get_local_tris(vertices, faces, device=device).reshape(-1, 2).double()
elif args.init == "tutte":
    inituv = torch.from_numpy(tutte_embedding(mesh.vertices, mesh.faces)).to(device) # F x 2
    inituv = inituv[faces].reshape(-1, 2)
elif args.init == "obj":
    inituv = torch.from_numpy(soup.uvs)
    inituv = inituv[faces].reshape(-1, 2)

objname = os.path.basename(datadir).split(".")[0]
if args.savename is None:
    screendir = os.path.join(savedir, objname)
else:
    screendir = os.path.join(savedir, args.savename)
Path(screendir).mkdir(parents=True, exist_ok=True)

#### Precompute the soft poisson parameters
grad = igl.grad(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy())
d_area = igl.doublearea(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy())
d_area = np.hstack((d_area, d_area, d_area)) # This matches the format for grad matrix (t0, t1, t2, ..., t0, t1, t2, ..., t0, t1, t2, ...)
mass = diags(d_area)
rhs = grad.T@mass
rhs = rhs.todense()
laplace = np.array((grad.T@mass@grad).todense())

## Update diagonal of Laplacian such that rows sum to 0
np.fill_diagonal(laplace, 0)
laplace[range(len(laplace)), range(len(laplace))] = -np.sum(laplace, axis=1)
np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)
grad = grad.todense()

## Convert to torch tensors
laplace = torch.from_numpy(laplace).double().to(device)
rhs = torch.from_numpy(rhs).double().to(device)

### Edge correspondences
from source_njf.utils import meshe_to_vpair
from source_njf.utils import get_edge_pairs, vertex_soup_correspondences, edge_soup_correspondences
from itertools import combinations
edgecorrespondences, facecorrespondences = edge_soup_correspondences(mesh.faces)
meshe_vpair_dict = meshe_to_vpair(mesh)
vpair_to_meshe = {v: k for k, v in meshe_vpair_dict.items()} # Reverse for vpair to meshe lookup

# Get corresponding edge lengths, og edge pairs, and face correspondences w/o boundary
elens_nobound = []
edge_vpairs = []
# NOTE: This maps eidx => new edge indexing with boundaries removed == initweights index!!!!
meshe_to_meshenobound = {}

ogedge_vpairs_nobound = []
facepairs_nobound = []
count = 0
for eidx, ogvpair in sorted(meshe_vpair_dict.items()):
    soupvpairs = edgecorrespondences[ogvpair]
    fpair = facecorrespondences[ogvpair]
    ogvpair = list(ogvpair)

    if len(soupvpairs) == 1:
        continue

    elens_nobound.append(np.linalg.norm(mesh.vertices[ogvpair[1]] - mesh.vertices[ogvpair[0]]))

    # Sanity check edge length
    np.testing.assert_almost_equal(elens_nobound[-1], mesh.length(mesh.topology.edges[eidx]))

    edge_vpairs.append(soupvpairs)
    meshe_to_meshenobound[eidx] = count
    ogedge_vpairs_nobound.append(ogvpair)

    # Get face pair
    assert len(fpair) == 2, f"Edge corresponding face pair {fpair} does not have 2 faces!"
    facepairs_nobound.append(fpair)

    count += 1

# NOTE: All are sorted by edges NOT on boundary!!
ogedge_vpairs_nobound = torch.tensor(ogedge_vpairs_nobound, device=device).long() # E x 2
elens_nobound = torch.tensor(elens_nobound, device=device)
facepairs_nobound = torch.tensor(facepairs_nobound, device=device).long()

# NOTE: FIRST square dimension gives the corresponding vertices across the two soup edges
edge_vpairs = np.array(edge_vpairs).transpose(0,2,1) # E x 2 x 2 (edges x (edge 1 v1, edge 1 v2) x (edge 2 v1, edge 2 v2)
edge_vpairs = torch.from_numpy(edge_vpairs).to(device).long()

if args.continuetrain:
    weights = torch.load(os.path.join(screendir, "weights.pt")).double().to(device)

    if args.no_weight_optimization:
        weights.requires_grad = False
    else:
        weights.requires_grad_()

    jacobians = torch.load(os.path.join(screendir, "jacobians.pt")).double().to(device)

    if args.no_jacobian_optimization:
        jacobians.requires_grad = False
    else:
        jacobians.requires_grad_()

    optim = torch.optim.Adam([weights, jacobians], lr=args.lr)

    try:
        with open(os.path.join(screendir, "epoch.pkl"), "rb") as f:
            starti = pickle.load(f)
    except Exception as e:
        print(e)
        starti = 0

    try:
        with open(os.path.join(screendir, "lossdict.pkl"), "rb") as f:
            lossdict = pickle.load(f)
    except Exception as e:
        print(e)
        lossdict = defaultdict(list)

    print(f"\n============ Continuing optimization from epoch {starti} ===========\n")
else:
    clear_directory(screendir)
    starti = 0
    lossdict = defaultdict(list)

    # Initial weights and jacobians
    weights = torch.ones(len(edge_vpairs), device=device, requires_grad=True, dtype=torch.double)

    if args.no_weight_optimization:
        weights.requires_grad = False

    jacobians = get_jacobian_torch(soupvs, soupfs, inituv, device=device).double()
    jacobians = torch.cat((jacobians, torch.zeros((jacobians.shape[0], 1, 3), device=device)), axis=1).double() # F x 3 x 3

    if args.no_jacobian_optimization:
        jacobians.requires_grad = False
    else:
        jacobians.requires_grad_()

    # Unit test: jacobians recover the original uvs up to translation per triangle
    with torch.no_grad():
        fverts = vertices[faces]
        pred_V = torch.einsum("abc,acd->abd", (fverts, jacobians[:,:2,:].transpose(2,1)))

        checkpre = inituv[soupfs,:2]
        diff = pred_V - checkpre
        diff -= torch.mean(diff, dim=1, keepdim=True) # Removes effect of per-triangle clobal translation
        torch.testing.assert_allclose(diff.float(), torch.zeros_like(diff), rtol=1e-6, atol=1e-6)

    optim = torch.optim.Adam([weights, jacobians], lr=args.lr)

framedir = os.path.join(screendir, "frames")
Path(framedir).mkdir(parents=True, exist_ok=True)

id = None
if args.continuetrain:
    import re
    if os.path.exists(os.path.join(screendir, 'wandb', 'latest-run')):
        for idfile in os.listdir(os.path.join(screendir, 'wandb', 'latest-run')):
            if idfile.endswith(".wandb"):
                result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                if result is not None:
                    id = result.group(1)
                    break
    else:
        print(f"Warning: No wandb record found in {os.path.join(screendir, 'wandb', 'latest-run')}!. Starting log from scratch...")

import wandb

c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
c.cleanup(int(1e9))

id = None
if args.continuetrain:
    import re
    if os.path.exists(os.path.join(savedir, args.savename, 'wandb', 'latest-run')):
        for idfile in os.listdir(os.path.join(savedir, args.savename, 'wandb', 'latest-run')):
            if idfile.endswith(".wandb"):
                result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                if result is not None:
                    id = result.group(1)
                    break
    else:
        print(f"Warning: No wandb record found in {os.path.join(savedir, args.savename, 'wandb', 'latest-run')}!. Starting log from scratch...")

wandb.login()
run = wandb.init(project='conetest_opt', name=args.savename, dir=os.path.join(savedir, args.savename),
                    mode= "offline" if args.debug else "online", id = id)
wandb.define_metric("iteration")
wandb.define_metric("loss", step_metric="iteration")
wandb.define_metric("stitchingloss", step_metric="iteration")
wandb.define_metric("araploss", step_metric="iteration")


from results_saving_scripts.plot_uv import plot_uv, export_views
import matplotlib.pyplot as plt

export_views(soupvs.detach().cpu().numpy(), soupfs.detach().cpu().numpy(), screendir, filename=f"mesh.png",
            plotname=f"starting mesh", n=1, cmap= plt.get_cmap("viridis"),
            fcolor_vals=np.arange(len(soupfs)), device="cpu", n_sample=100, width=400, height=400,
            vmin=0, vmax=len(soupfs), shading=True)
plot_uv(screendir, f"inituv", inituv.detach().cpu().numpy(),  soupfs.detach().cpu().numpy(), losses=None,
        facecolors = np.arange(len(soupfs))/(len(soupfs)-1))

# Initialize random weights
from source_njf.losses import vertexseparation, symmetricdirichlet, uvgradloss, arap
from source_njf.utils import get_jacobian_torch, clear_directory, dclamp

for iteration in range(starti, args.niters):
    loss = 0

    # Weights must be nonzero
    weights = dclamp(weights, 1e-10, 1e6)

    ## Recompute the RHS based on new jacobians
    input = jacobians.transpose(2, 1).reshape(1, -1, 3) # 1 x F*3 x 3

    # Reshape the Jacobians to match the format of grad (vertex ordering STAYS THE SAME)
    P = torch.zeros(input.shape, device=device, dtype=rhs.dtype)
    k = input.shape[1] // 3
    P[:, :k, :] = input[:, ::3] # First row of all jacobians together
    P[:, k:2 * k, :] = input[:, 1::3] # Second row of all jacobians together
    P[:, 2 * k:, :] = input[:, 2::3] # Third row of all jacobians together
    input_to_solve = rhs @ P

    sp_idxs = edge_vpairs.reshape(-1, 2) # E * 2 x 2 (last dimension compares corresponding vertices (two pairs per edge))
    sp_idxs = (torch.cat([sp_idxs[:,0],sp_idxs[:,1]]), torch.cat([sp_idxs[:,1],sp_idxs[:,0]]))

    sp_weights = weights.repeat_interleave(2)
    sp_weights = torch.cat([sp_weights, sp_weights])

    templaplace = laplace.clone().detach()
    templaplace[sp_idxs] = sp_weights # F*3 x F*3

    # Pin vertices
    pinlaplace = templaplace
    pin_input = input_to_solve

    # Diagonal needs to be equal to sum of pair weights for each row
    pinlaplace.fill_diagonal_(0)
    pinlaplace[range(len(pinlaplace)), range(len(pinlaplace))] = -torch.sum(pinlaplace, dim=1)
    torch.testing.assert_close(torch.sum(pinlaplace, dim=1), torch.zeros(pinlaplace.shape[0], device=device, dtype=pinlaplace.dtype), atol=1e-4, rtol=1e-4)

    # Solve with pinned vertices
    uvs = torch.linalg.solve(pinlaplace[1:,1:], pin_input[0, 1:]) # F*3 x 3
    uvs = uvs[:,:2]

    # Add back pinned vertex
    full_uvs = torch.cat([torch.zeros(1, 2, device=device), uvs], dim=0)

    # Face UVs
    uv_face = full_uvs.reshape(-1, 3, 2) # F x 3 x 2

    # Center the UVs at origin
    from source_njf.utils import normalize_uv

    vizuv = uv_face.clone()
    with torch.no_grad():
        normalize_uv(vizuv)

        # Scale back to centered at 0.5
        vizuv += 0.5

    # SDS loss
    # Prereqs: texture image, texture description
    if args.sdsloss:
        assert args.textureimg is not None and args.texturetext is not None, "Need to specify textureimg and texturetext for SDS loss"
        from source_njf.diffusion_guidance.deepfloyd_if import DeepFloydIF, DeepFloydIF_Img2Img
        from source_njf.diffusion_guidance.deepfloyd_cascaded import DeepFloydCascaded, DiffusionConfig

        if args.sdsloss == "text2img":
            diffusion = DeepFloydIF() # optionally you can pass a config at initialization
        elif args.sdsloss == "img2img":
            diffusion = DeepFloydIF_Img2Img()
        elif args.sdsloss == "cascaded":
            cfg = DiffusionConfig()
            diffusion = DeepFloydCascaded(cfg)

        # Text encoding
        texturetext = ' '.join(args.texturetext)
        text_z, text_z_neg = diffusion.encode_prompt(texturetext)

        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        textureimg = pil_to_tensor(Image.open(args.textureimg)).double().to(device)

        rgb_images = []
        # NOTE: Can upscale resolution to get better gradients
        from source_njf.renderer import render_texture
        num_views = 5
        radius = 2.5
        center = torch.zeros(2)
        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
            :-1].double().to(device)
        elev = torch.zeros(len(azim), device=device).double()

        rgb_images.append(render_texture(vertices.double(), faces, vizuv, elev, azim, radius, textureimg/255, lights=None,
                                                resolution=(args.resolution, args.resolution), device=device, lookatheight=0, whitebg=True,
                                                interpolation_mode='bilinear'))
        ## Debugging
        # import matplotlib.pyplot as plt

        # images = rgb_images[0]['image'].detach().cpu().numpy()
        # num_views = 5
        # fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
        # for nview in range(num_views):
        #     j = nview % 5
        #     if nview > 5:
        #         i = nview // 5
        #         axs[i,j].imshow(images[nview].transpose(1,2,0))
        #         axs[i,j].axis('off')
        #     else:
        #         axs[j].imshow(images[nview].transpose(1,2,0))
        #         axs[j].axis('off')
        # plt.axis('off')
        # fig.suptitle(f"Current Textures")
        # plt.savefig(os.path.join(framedir, f"test.png"))
        # plt.close(fig)
        # plt.cla()

        if args.sdsloss == "text2img":
            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z)
            sdsloss = sds['loss_sds']
            loss += sdsloss
            lossdict['SDS Loss'].append(sdsloss.item())
        elif args.sdsloss == "cascaded":
            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z, stage = 'I')
            sdsloss = sds['loss_sds']
            loss += sdsloss
            lossdict['SDS Stage 1'].append(sdsloss.item())

            sds = diffusion(rgb_images[0]['image'], prompt_embeds = text_z, stage = 'II')
            sdsloss = sds['loss_sds']
            loss += args.stage2weight * sdsloss
            lossdict['SDS Stage 2'].append(sdsloss.item())

    if args.imageloss:
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        texturename = os.path.basename(args.textureimg).split(".")[0]
        img = Image.open(args.textureimg)
        img = img.convert("RGB")
        textureimg = pil_to_tensor(img).double().to(device)

        rgb_images = []

        from source_njf.renderer import render_texture
        total_views = 5
        num_views = args.imagelossn
        radius = 2.5
        center = torch.zeros(2)
        azim = torch.linspace(center[0], 2 * np.pi + center[0], total_views + 1)[
            :-1].double().to(device)
        elev = torch.zeros(len(azim), device=device).double()

        # Subset to the number of views input
        # azim = azim[[3]]
        # elev = elev[[3]]

        pred_images = render_texture(vertices.double(), faces, vizuv, elev, azim, radius, textureimg/255, lights=None,
                                                resolution=(args.renderresolution, args.renderresolution), device=device, lookatheight=0, whitebg=True,
                                                interpolation_mode = args.interpmode)
        pred_images = Resize((args.resolution, args.resolution), antialias=True)(pred_images['image'])

        gt_images = []

        # HACK: we only look at the 3rd render for flat triangle
        for i in range(num_views):
            gt_image = torchvision.io.read_image(args.imageloss + f"_{i}.png").double().to(device)
            gt_image = Resize((args.resolution, args.resolution))(gt_image)/255
            gt_images.append(gt_image)
        # gt_image = torchvision.io.read_image(args.imageloss + f"_3.png").double().to(device)
        # gt_image = Resize((args.resolution, args.resolution))(gt_image)/255
        # gt_images.append(gt_image)

        gt_images = torch.stack(gt_images, dim=0)
        imageloss = torch.nn.functional.mse_loss(pred_images, gt_images, reduction="none")
        loss += torch.mean(imageloss)
        lossdict['Image Loss'].append(imageloss.cpu().detach().numpy())

    if args.distortionloss:
        from source_njf.losses import arap
        araploss = arap(ogvertices, ogfaces, uv_face, paramtris=uv_face, device=device)
        loss += args.distortionloss_weight * torch.mean(araploss)
        lossdict['Distortion Loss'].append(args.distortionloss_weight * araploss.detach().cpu().numpy())

    if args.edgecutloss:
        uv = uv_face.reshape(-1, 2)
        uvpairs = uv[edge_vpairs] # E x 2 x 2 x 2
        elens = elens_nobound

        ## Edge separation loss
        edgesep = torch.sum(torch.nn.functional.mse_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2) # E x 2
        edgesep = torch.mean(edgesep, dim=1) # E x 1

        # Weight with edge lengths
        wedgesep = edgesep * elens

        if args.seamlessedgecut:
            edgecutloss = wedgesep * wedgesep/(wedgesep * wedgesep + args.seamlessdelta)
        else:
            edgecutloss = wedgesep

        loss += args.edgecutloss_weight * torch.mean(edgecutloss)
        lossdict['Edge Cut Loss'].append(args.edgecutloss_weight * edgecutloss.detach().cpu().numpy())

    # Compute loss
    loss.backward()
    lossdict['Total Loss'].append(loss.item())

    optim.step()
    optim.zero_grad()

    print(f"========== Done with iteration {iteration}. ==========")
    lossstr = ""
    for k, v in lossdict.items():
        lossstr += f"{k}: {np.mean(v[-1]):0.7f}. "
    print(lossstr)

    wandb.log({'total_loss': lossdict['Total Loss'][-1]}, step=iteration, commit=True)
    if args.edgecutloss:
        wandb.log({'stitching_loss': np.mean(lossdict['Edge Cut Loss'][-1])}, step=iteration, commit=True)
    if args.distortionloss:
        wandb.log({'arap_loss': np.mean(lossdict['Distortion Loss'][-1])}, step=iteration, commit=True)
    if args.sdsloss:
        if args.sdsloss == "cascaded":
            wandb.log({'sds_loss_1': lossdict['SDS Stage 1'][-1]}, step=iteration, commit=True)
            wandb.log({'sds_loss_2': lossdict['SDS Stage 2'][-1]}, step=iteration, commit=True)
        else:
            wandb.log({'sds_loss': lossdict['SDS Loss'][-1]}, step=iteration, commit=True)
    if args.imageloss:
        wandb.log({'image_loss': np.mean(lossdict['Image Loss'][-1])}, step=iteration, commit=True)

    ogvs = ogvertices.detach().cpu().numpy()
    ogfs = ogfaces.detach().cpu().numpy()
    ogvsoup = ogvs[ogfs].reshape(-1, 3)

    # Visualize every viziter epochs
    if iteration % args.viziter == 0:
        # Convert loss dict keys
        lossdict_convert = {}
        if args.edgecutloss:
            lossdict_convert['Edge Cut Loss'] = 'edgecutloss'
        if args.distortionloss:
            lossdict_convert['Distortion Loss'] = 'distortionloss'

        lossdict_viz = {}
        for key, item in lossdict.items():
            if key in lossdict_convert:
                lossdict_viz[lossdict_convert[key]] = item[-1]

        plot_uv(framedir, f"uv_{iteration:06}", uv_face.reshape(-1, 2).detach().cpu().numpy(), soupfs.detach().cpu().numpy(),
                losses=lossdict_viz, edge_vpairs = edge_vpairs.detach().cpu().numpy(),
                facecolors = np.arange(len(soupfs))/(len(soupfs)-1))

        imagepaths = [os.path.join(framedir, f"uv_{iteration:06}.png"), args.imageloss + f"_UV.png"] + \
                    [os.path.join(framedir, f"{key}_uv_{iteration:06}.png") for key in lossdict_viz.keys() if "loss" in key]
        images = [wandb.Image(Image.open(x)) for x in imagepaths]
        wandb.log({'uvs': images}, commit=True)

        if args.sdsloss or args.imageloss:
            import matplotlib.pyplot as plt

            images = pred_images.detach().cpu().numpy()
            num_views = len(images)
            fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
            if num_views == 1:
                axs.imshow(images[0].transpose(1,2,0), aspect='equal')
                axs.axis('off')
            else:
                for nview in range(num_views):
                    j = nview % 5
                    if nview > 5:
                        i = nview // 5
                        axs[i,j].imshow(images[nview].transpose(1,2,0), aspect='equal')
                        axs[i,j].axis('off')
                    else:
                        axs[j].imshow(images[nview].transpose(1,2,0), aspect='equal')
                        axs[j].axis('off')
            plt.axis('off')
            fig.suptitle(f"Epoch {iteration} Textures")
            plt.savefig(os.path.join(framedir, f"texture_{iteration:06}.png"))
            plt.close()
            plt.cla()

            # Log the plotted imgs
            images = [wandb.Image(Image.open(os.path.join(framedir, f"texture_{iteration:06}.png"))), wandb.Image(Image.open(args.imageloss + f".png"))]
            wandb.log({'textures': images}, commit=True)

        # Plot the image loss image
        if args.imageloss:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots()
            imageloss = (lossdict['Image Loss'][0] * 255).astype(np.uint8).transpose(0,2,3,1)
            num_views = imageloss.shape[0]
            fig, axs = plt.subplots(int(np.ceil(num_views/5)), num_views)
            if num_views == 1:
                axs.imshow(Image.fromarray(imageloss[0]))
                axs.axis('off')
            else:
                for nview in range(num_views):
                    j = nview % 5
                    if nview > 5:
                        i = nview // 5
                        axs[i,j].imshow(Image.fromarray(imageloss[nview]))
                        axs[i,j].axis('off')
                    else:
                        axs[j].imshow(Image.fromarray(imageloss[nview]))
                        axs[j].axis('off')
            plt.axis('off')
            fig.suptitle(f"Epoch {iteration} Image Loss")
            plt.savefig(os.path.join(framedir, f"imageloss_{iteration:06}.png"))
            plt.close()
            plt.cla()

            # Log the plotted imgs
            images = [wandb.Image(Image.open(os.path.join(framedir, f"imageloss_{iteration:06}.png")))]
            wandb.log({'imageloss': images}, commit=True)

        # Weights
        import matplotlib.pyplot as plt

        plt.hist(weights.detach().cpu().numpy(), bins=100)
        plt.title(f"Weights Histogram")
        plt.savefig(os.path.join(framedir, f"weights_{iteration:06}.png"))
        plt.close()
        plt.cla()

        torch.save(weights.detach().cpu(), os.path.join(screendir, "weights.pt"))
        torch.save(jacobians.detach().cpu(), os.path.join(screendir, "jacobians.pt"))

        with open(os.path.join(screendir, "epoch.pkl"), 'wb') as f:
            pickle.dump(iteration, f)

        with open(os.path.join(screendir, "lossdict.pkl"), "wb") as f:
            pickle.dump(lossdict, f)

# Plot loss curves
# import matplotlib.pyplot as plt
# for k, v in lossdict.items():
#     fig, axs = plt.subplots()
#     axs.plot(np.arange(len(v)), v)
#     axs.set_title(k)

#     lossname = k.replace(" ", "_").lower()
#     plt.savefig(os.path.join(screendir, f"{lossname}.png"))
#     plt.cla()

# Save final UVs, weights, and jacobians
np.save(os.path.join(screendir, "uv.npy"), uv_face.reshape(-1, 2).detach().cpu().numpy())
np.save(os.path.join(screendir, "weights.npy"), weights.detach().cpu().numpy())
np.save(os.path.join(screendir, "jacobians.npy"), jacobians.detach().cpu().numpy())

# Show flipped triangles
from source_njf.utils import get_flipped_triangles
flipped = get_flipped_triangles(uv_face.reshape(-1, 2).detach().cpu().numpy(), soupfs.detach().cpu().numpy())
flipvals = np.zeros(len(soupfs))
flipvals[flipped] = 1
lossdict = {'fliploss': flipvals}

plot_uv(screendir, f"finaluv", uv_face.reshape(-1, 2).detach().cpu().numpy(), soupfs.detach().cpu().numpy(),
                    facecolors = np.arange(len(soupfs))/(len(soupfs)-1),
                    losses=lossdict)