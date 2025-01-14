import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.tri import Triangulation
import numpy as np
import os
import fresnel

def plot_uv(path, name, pred_vertices, triangles, gt_vertices=None, losses=None, cmin=0, cmax=2,
            stitchweight=0.1, dmin=0, dmax=1,
            facecolors = None, edges = None, edgecolors = None, source = None, valid_edges_to_soup = None,
            cmap = plt.get_cmap("tab20"), edge_cmap=plt.get_cmap("gist_rainbow"), edgecorrespondences=None,
            keepidxs=None,
            ):
    """ edges: E x 2 x 2 array of individiual node positions
        edgecolors: [0-1] values to query edge_cmap """
    # First center the predicted vertices
    # pred_vertices -= np.mean(pred_vertices, 0, keepdims=True) # Sum batched over faces, vertex dimension
    tris = Triangulation(pred_vertices[:, 0], pred_vertices[:, 1], triangles=triangles)

    # setup side by side plots
    if facecolors is None:
        facecolors=np.ones(len(triangles)) * 0.5

    fname = "_".join(name.split())
    if gt_vertices:
        fig, axs = plt.subplots(1,2,figsize=(15, 4))
        fig.suptitle(name)

        # plot GT
        axs[0].set_title('GT')
        axs[0].axis('equal')

        gt_tris = Triangulation(gt_vertices[:,0], gt_vertices[:,1], triangles)
        axs[0].triplot(gt_tris, linewidth=0.1)

        # plot ours
        axs[1].set_title('Ours')
        axs[1].axis('equal')
        # axs[1].set_axis_off()

        axs[1].triplot(tris, linewidth=0.1)
        plt.axis('off')
        plt.savefig(os.path.join(path, f"{fname}.png"))
        plt.close(fig)
        plt.cla()
    else:
        fig, axs = plt.subplots(figsize=(5, 5))
        # plot ours
        axs.set_title(name)
        axs.tripcolor(tris, facecolors=facecolors, cmap=cmap,
                            linewidth=0.1, edgecolor="black")

        # Plot edges if given
        if edges is not None and len(edges) > 0:
            for i, e in enumerate(edges):
                axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                         color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

        plt.axis('off')
        axs.axis('equal')
        plt.savefig(os.path.join(path, f"{fname}.png"))
        plt.close(fig)
        plt.cla()

    # # NOTE: this only works with WandbLogger
    # if logger is not None:
    #     logger.experiment.log({fname: wandb.Image(os.path.join(path, f"{fname}.png"))})

    # Plot losses
    if losses is not None:
        for key, val in losses.items():
            if "loss" in key: # Hacky way of avoiding aggregated values
                if key == "vertexseploss":
                    if source is None:
                        print(f"Need to pass in source to plot {key}")
                        continue

                    valid_pairs = source.valid_pairs
                    separationloss = val

                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"Total Vertex Loss: {np.sum(separationloss):0.8f}")
                    cmap = plt.get_cmap("Reds")

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
                    vseplosses = np.zeros(len(pred_vertices))
                    for k, v in sorted(vlosses.items()):
                        vseplosses[k] = v / vlosscount[k]

                    axs.tripcolor(tris, vseplosses, cmap=cmap, shading='gouraud',
                    linewidth=0.5, vmin=0, vmax=0.2, edgecolor='black')

                    # Plot edges if given
                    if edges is not None and len(edges) > 0:
                        for i, e in enumerate(edges):
                            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                        color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight', dpi=600)
                    plt.close()

                elif key == "edgecutloss":
                    from collections import defaultdict

                    if source is None:
                        print(f"Need to pass in source mesh to plot {key}")
                        continue

                    edgecutloss = val # E x 1
                    edge_vpairs = source.edge_vpairs
                    if keepidxs is not None:
                        edge_vpairs = edge_vpairs[keepidxs]
                    assert len(edgecutloss) == len(edge_vpairs), f"len(edgecutloss) {len(edgecutloss)} != len(edge_vpairs) {len(edge_vpairs)}"

                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"Total Edge Cut Loss: {np.sum(edgecutloss):0.8f}")

                    cmap = plt.get_cmap("Reds")
                    norm = mpl.colors.Normalize(vmin=0, vmax=stitchweight)
                    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

                    # Plot color-coded triangles and color the edges based on losses
                    facecmap = plt.get_cmap("tab20")
                    axs.tripcolor(tris, facecolors=facecolors, cmap=facecmap,
                                    linewidth=0.5, edgecolor='black')

                    # Plot edges if given
                    for i in range(len(edgecutloss)):
                        edgeval = edgecutloss[i]
                        uv0 = pred_vertices[edge_vpairs[i][:,0]]
                        uv1 = pred_vertices[edge_vpairs[i][:,1]]

                        axs.plot(uv0[:, 0], uv0[:, 1], marker='none', linestyle='-',
                                    color=scalarmap.to_rgba(edgeval)[:3], linewidth=1.5)

                        axs.plot(uv1[:, 0], uv1[:, 1], marker='none', linestyle='-',
                                    color=scalarmap.to_rgba(edgeval)[:3], linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight', dpi=600)
                    plt.close()

                elif key == "edgegradloss": # E x 2
                    if edgecorrespondences is None:
                        print(f"Need to pass in edge correspondences to plot {key}")
                        continue

                    uve1 = []
                    uve2 = []
                    for k, v in sorted(edgecorrespondences.items()):
                        # If only one correspondence, then it is a boundary
                        if len(v) == 1:
                            continue
                        uve1.append(pred_vertices[list(v[0])])
                        uve2.append(pred_vertices[list(v[1])])

                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"Total Edge Grad Loss: {np.sum(val):0.8f}")
                    cmap = plt.get_cmap("Reds")
                    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
                    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
                    ecolors = scalarmap.to_rgba(val)[:,:3]

                    # Plot monotone triangles and color the edges based on losses
                    axs.tripcolor(tris, facecolors=np.ones(len(triangles)) * 0.5, cmap=cmap,
                                    linewidth=0.5, edgecolor='black')

                    # Plot edges if given
                    for i, e in enumerate(uve1):
                        axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                    color=ecolors[i], linewidth=1.5)

                    for i, e in enumerate(uve2):
                        axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                    color=ecolors[i], linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight', dpi=600)
                    plt.close()
                elif key == "distortionloss":
                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"{name}\nAvg {key}: {np.mean(val):0.4f}")
                    cmap = plt.get_cmap("Reds")
                    axs.tripcolor(tris, val[:len(triangles)], facecolors=val[:len(triangles)], cmap=cmap,
                                linewidth=0.1, vmin=dmin, vmax=dmax, edgecolor="black")

                    # Plot edges if given
                    if edges is not None or len(edges) > 0:
                        for i, e in enumerate(edges):
                            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                    color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight',dpi=600)
                    plt.close()
                elif key == "invjloss":
                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"{name}\nAvg {key}: {np.mean(val):0.4f}")
                    cmap = plt.get_cmap("Reds")
                    axs.tripcolor(tris, val[:len(triangles)], facecolors=val[:len(triangles)], cmap=cmap,
                                linewidth=0.1, vmin=cmin, vmax=cmax, edgecolor="black")

                    # Plot edges if given
                    if edges is not None or len(edges) > 0:
                        for i, e in enumerate(edges):
                            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                    color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight',dpi=600)
                    plt.close()
                elif key == "fliploss":
                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"{name}\nAvg {key}: {np.mean(val):0.4f}")
                    cmap = plt.get_cmap("Reds")
                    axs.tripcolor(tris, val[:len(triangles)], facecolors=val[:len(triangles)], cmap=cmap,
                                linewidth=0.1, vmin=cmin, vmax=cmax, edgecolor="black")

                    # Plot edges if given
                    if edges is not None or len(edges) > 0:
                        for i, e in enumerate(edges):
                            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                    color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight',dpi=600)
                    plt.close()
                elif key == "gtuvloss":
                    fig, axs = plt.subplots(figsize=(5,5))
                    fig.suptitle(f"Total GT Loss: {np.sum(val):0.8f}")
                    cmap = plt.get_cmap("Reds")

                    axs.tripcolor(tris, val, cmap=cmap,
                                linewidth=0.1, vmin=0, vmax=1, edgecolor='black')

                    # Plot edges if given
                    if edges is not None and len(edges) > 0:
                        for i, e in enumerate(edges):
                            axs.plot(e[:, 0], e[:, 1], marker='none', linestyle='-',
                                        color=edge_cmap(edgecolors[i]) if edgecolors is not None else "black", linewidth=1.5)

                    plt.axis('off')
                    axs.axis("equal")
                    plt.savefig(os.path.join(path, f"{key}_{fname}.png"), bbox_inches='tight', dpi=600)
                    plt.close()
                else:
                    print(f"No visualization code for {key}")
                    continue


def export_views(vertices, faces, savedir, n=5, n_sample=20, width=150, height=150, plotname="Views", filename="test", fcolor_vals=None,
                 vcolor_vals=None, shading=True, cylinders=None, cylinder_scalars=None, subcylinders=None,
                 device="cpu", outline_width=0.005, cmap= plt.get_cmap("Reds"), vmin=0, vmax=1):
    import torch
    import matplotlib as mpl
    from matplotlib import cm

    fresnel_device = fresnel.Device(mode=device)
    scene = fresnel.Scene(device=fresnel_device)

    # Create cylinders (edges)
    # cylinders: (N, 2, 3) where every row is endpoints of cylinders
    if cylinders is not None:
        geometry = fresnel.geometry.Cylinder(scene, N=len(cylinders))
        geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.5,0,0]),
                                                    roughness=1)
        geometry.points[:] = cylinders
        geometry.radius[:] = np.ones(len(cylinders)) * outline_width * 2
        geometry.material.solid = 1.0

        if cylinder_scalars is not None: # Should be N x 2
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

            cylinder_colors = scalarmap.to_rgba(cylinder_scalars)[:,:,:3] # N x 2 x 3
            geometry.color[:] = cylinder_colors
            geometry.material.primitive_color_mix = 1.0

    if subcylinders is not None:
        geometry = fresnel.geometry.Cylinder(scene, N=len(subcylinders))
        geometry.material = fresnel.material.Material(color=fresnel.color.linear([0,0,1]),
                                                    roughness=1)
        geometry.points[:] = subcylinders
        geometry.radius[:] = np.ones(len(subcylinders)) * outline_width * 2
        geometry.material.solid = 1.0

    # Create mesh
    fverts = vertices[faces].reshape(3 * len(faces), 3)
    mesh = fresnel.geometry.Mesh(scene, vertices=fverts, N=1)
    mesh.material = fresnel.material.Material(color=fresnel.color.linear([0.25, 0.5, 0.9]), roughness=0.1)

    # NOTE: outline width 0 => outline off
    mesh.outline_material = fresnel.material.Material(color=(0., 0., 0.), roughness=0.1, solid=1)
    mesh.outline_width=outline_width

    # Shading
    if not shading:
        mesh.material.solid = 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    if fcolor_vals is not None:
        # Each face gets own set of 3 vertex colors
        fcolors = scalarmap.to_rgba(fcolor_vals)[:,:3]
        vcolors = [color for color in fcolors for _ in range(3)]
        vcolors = np.vstack(vcolors)

        mesh.color[:] = fresnel.color.linear(vcolors)
        mesh.material.primitive_color_mix = 1.0
    elif vcolor_vals is not None:
        vcolors = scalarmap.to_rgba(vcolor_vals)[:,:3]
        mesh.color[:] = fresnel.color.linear(vcolors)
        mesh.material.primitive_color_mix = 1.0

    scene.lights = fresnel.light.cloudy()

    # TODO: maybe initializing with fitting gives better camera angles
    scene.camera = fresnel.camera.Orthographic.fit(scene, margin=0)
    # Radius is just largest vertex norm
    r = np.max(np.linalg.norm(vertices))
    elevs = torch.linspace(0, 2 * np.pi, n+1)[:n]
    azims = torch.linspace(-np.pi, np.pi, n+1)[:n]
    renders = []
    for i in range(len(elevs)):
        elev = elevs[i]
        azim = azims[i]
        # Then views are just linspace
        # Loop through all camera angles and collect outputs
        pos, lookat, _ = get_camera_from_view(elev, azim, r=r)
        scene.camera.look_at = lookat
        scene.camera.position = pos
        out = fresnel.pathtrace(scene, samples=n_sample, w=width,h=height)
        renders.append(out[:])

    # Plot and save in matplotlib using imshow
    import matplotlib.pyplot as plt
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig, axs = plt.subplots(nrows=1, ncols=len(renders), gridspec_kw={'wspace':0, 'hspace':0}, figsize=(15, 4), squeeze=True)
    for i in range(len(renders)):
        render = renders[i]
        if len(renders) == 1:
            axs.set_xticks([])
            axs.set_yticks([])
            axs.imshow(render, interpolation='lanczos')
        else:
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].imshow(render, interpolation='lanczos')
    fig.suptitle(plotname)
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, filename))
    plt.cla()
    plt.close()

# NOTE: This assumes default view direction of (0, 0, -r)
def get_camera_from_view(elev, azim, r=2.0):
    x = r * np.cos(azim) *  np.sin(elev)
    y = r * np.sin(azim) * np.sin(elev)
    z = r * np.cos(elev)

    pos = np.array([x, y, z])
    look_at = -pos
    direction = np.array([0.0, 1.0, 0.0])
    return pos, look_at, direction