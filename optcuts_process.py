### OptCuts: run same mesh a bunch of times and check minimal energy result
from meshing.io import PolygonSoup
from meshing.mesh import Mesh
import subprocess
import os
import argparse
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--meshpath", type=str, required=True)
argparser.add_argument("--savepath", type=str, required=True)
argparser.add_argument("--i", type=int, required=True)
args = argparser.parse_args()
i = args.i
meshpath = args.meshpath
savedir = args.savepath
Path(savedir).mkdir(parents=True, exist_ok=True)

def run_optcuts(meshpath, i):
    result = subprocess.call(["./OptCuts_bin", "100", meshpath,
                                "0.999", f"{i}", "0", "4.05", "1", "0"], stdout=subprocess.DEVNULL)
    return result

from collections import defaultdict
import numpy as np
from source_njf.losses import symmetricdirichlet
from source_njf.utils import get_jacobian_torch
import torch

edgevpairs = defaultdict(int)

run_optcuts(meshpath, i)
meshname = os.path.splitext(os.path.basename(meshpath))[0]
cutpath = f"./output/{meshname}_Tutte_0.999_{i}_OptCuts/finalResult_mesh.obj"
if not os.path.exists(cutpath):
    cutpath = f"./output/{meshname}_HighGenus_0.999_{i}_OptCuts/finalResult_mesh.obj"

cutsoup = PolygonSoup.from_obj(cutpath)
cutuvmesh = Mesh(cutsoup.uvs, cutsoup.face_uv)
fuv_idxs = cutsoup.face_uv
fuv_to_v = np.zeros(np.max(fuv_idxs) + 1).astype(int)
fuv_to_v[fuv_idxs.flatten()] = cutsoup.indices.flatten()

for boundary in cutuvmesh.topology.boundaries.values():
    cutvs = [v.index for v in boundary.adjacentVertices()]
    cutpairs = [frozenset([fuv_to_v[cutvs[i]], fuv_to_v[cutvs[i+1]]]) for i in range(len(cutvs)-1)] + [frozenset([fuv_to_v[cutvs[-1]], fuv_to_v[cutvs[0]]])]

    for pair in cutpairs:
        edgevpairs[pair] += 1

# Compute distortion
# uv = cutsoup.uvs
# uvfs = torch.from_numpy(uv[cutsoup.face_uv].reshape(-1, 2))
# soupvs = torch.from_numpy(cutsoup.vertices[cutsoup.indices].reshape(-1, 3))

# js = get_jacobian_torch(soupvs, torch.arange(len(soupvs)).reshape(-1, 3), uvfs)
# distortion = symmetricdirichlet(soupvs, torch.arange(len(soupvs)).reshape(-1, 3), js).mean().item()

# Get energies
outdir = f"./output/{meshname}_Tutte_0.999_{i}_OptCuts"

with open(os.path.join(outdir, "energyValPerIter.txt"), 'r') as f:
    for line in f:
        pass
    energies = line.split()

denergy = float(energies[1])
cutenergy = float(energies[2])

print(f"Distortion: {denergy}")
print(f"Cut energy: {cutenergy}")
print(f"Boundary size: {len(edgevpairs)}")

import dill as pickle

with open(f"{savedir}/distortion.pkl", "wb") as f:
    pickle.dump(denergy, f)

with open(f"{savedir}/cutenergy.pkl", "wb") as f:
    pickle.dump(cutenergy, f)

with open(f"{savedir}/edgevotes.pkl", "wb") as f:
    pickle.dump(edgevpairs, f)

# Copy finalresult mesh to savedir
import shutil
shutil.copy(cutpath, f"{savedir}/{meshname}_{i}_cut.obj")