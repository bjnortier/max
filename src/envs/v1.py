import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

model = load_model_from_path('../skilos/src/envs/assets/metalhead_v2.xml')
sim = MjSim(model)
print(sim.data.qpos)
viewer = MjViewer(sim)
while True:
    # sim.step()
    viewer.render()
