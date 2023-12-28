# Render GT of cylinder texture
from source_njf.diffusion_guidance.deepfloyd_if import DeepFloydIF, DeepFloydIF_Img2Img
from source_njf.diffusion_guidance.deepfloyd_cascaded import DeepFloydCascaded, DiffusionConfig

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch
import numpy as np
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

textures = ["./data/textures/discretemap.png", "./data/textures/discreterainbow.jpg",
            "./data/textures/speckledgradient.png", "./data/textures/checkerboard.jpeg"]
datadirs = ["./data/cylinder/cylinder.obj"]

for texture in textures:
    for datadir in datadirs:
        texturename = os.path.basename(texture).split(".")[0]
        img = Image.open(texture)
        img = img.convert("RGB")
        textureimg = pil_to_tensor(img).double().to(device)
        rgb_images = []

        # NOTE: Can upscale resolution to get better gradients
        from source_njf.renderer import render_texture
        num_views = 5
        radius = 2.5
        center = torch.zeros(2)
        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
            :-1].double().to(device)
        elev = torch.zeros(len(azim)).double().to(device)

        # Face UVs
        from meshing.io import PolygonSoup
        from meshing.mesh import Mesh

        meshname = os.path.basename(datadir).split(".")[0]
        soup = PolygonSoup.from_obj(datadir)
        mesh = Mesh(soup.vertices, soup.indices)
        mesh.normalize()
        vertices = torch.from_numpy(mesh.vertices).to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)

        # Compute GT UVs
        from source_njf.utils import SLIM
        uvs = torch.from_numpy(SLIM(mesh)[0]).double().to(device)
        uv_face = uvs[faces].reshape(-1, 3, 2).to(device)

        # Center the UVs at origin
        from source_njf.utils import normalize_uv
        with torch.no_grad():
            normalize_uv(uv_face)

        # Rotate using rotmatrix
        theta = torch.tensor([np.pi/3]).double()
        rotmatrix = torch.cat([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)]).reshape(2, 2).to(device)
        uv_face = uv_face @ rotmatrix

        # Scale back to centered at 0.5
        uv_face += 0.5

        with torch.no_grad():
            rgb_images.append(render_texture(vertices.double(), faces, uv_face, elev, azim, radius, textureimg/255, lights=None,
                                                    resolution=(512, 512), lookatheight=0, whitebg=True,
                                                    interpolation_mode='bilinear', device=device))

        import matplotlib.pyplot as plt

        images = rgb_images[0]['image'].detach().cpu().numpy()
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
        fig.suptitle(f"Current Textures")
        plt.savefig(f"./outputs/{meshname}_{texturename}_gt.png")
        plt.close()
        plt.cla()

        # Save ground truth images
        for nview in range(num_views):
            img = Image.fromarray((images[nview].transpose(1,2,0) * 255).astype(np.uint8))
            img.save(f"./outputs/{meshname}_{texturename}_gt_{nview}.png")

        # Plot UVs
        from results_saving_scripts.plot_uv import plot_uv
        plot_uv("./outputs", f"{meshname}_{texturename}_gt_UV", uv_face.reshape(-1, 2).detach().cpu().numpy(), np.arange(len(uv_face.reshape(-1, 2))).reshape(-1, 3))