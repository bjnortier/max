import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from math import pi

model = load_model_from_path('../skilos/src/envs/assets/metalhead_v4.xml')
sim = MjSim(model)

# print('torso:x', sim.model.get_joint_qpos_addr("torso:x"))
# print('torso:pitch', sim.model.get_joint_qpos_addr("torso:pitch"))
# print('torso:roll', sim.model.get_joint_qpos_addr("torso:roll"))

#
# print('back_right_hip', sim.model.get_joint_qpos_addr('back_right_hip'))
# print('back_right_knee', sim.model.get_joint_qpos_addr('back_right_knee'))
# print('back_right_ankle', sim.model.get_joint_qpos_addr('back_right_ankle'))
# print('back_left_hip', sim.model.get_joint_qpos_addr('back_left_hip'))
# print('back_left_knee', sim.model.get_joint_qpos_addr('back_left_knee'))
# print('back_left_ankle', sim.model.get_joint_qpos_addr('back_left_ankle'))
# print('front_right_hip', sim.model.get_joint_qpos_addr('front_right_hip'))
# print('front_right_knee', sim.model.get_joint_qpos_addr('front_right_knee'))
# print('front_right_ankle', sim.model.get_joint_qpos_addr('front_right_ankle'))
# print('front_left_hip', sim.model.get_joint_qpos_addr('front_left_hip'))
# print('front_left_knee', sim.model.get_joint_qpos_addr('front_left_knee'))
# print('front_left_ankle', sim.model.get_joint_qpos_addr('front_left_ankle'))
#
# sim_state = sim.get_state()
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_right_hip')] = -19.41/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_right_knee')] = 50.66/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_right_ankle')] = -57.97/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_left_hip')] = -19.41/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_left_knee')] = 50.66/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('back_left_ankle')] = -57.97/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_right_hip')] = 13.74/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_right_knee')] = -38.98/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_right_ankle')] = 37.90/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_left_hip')] = 13.74/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_left_knee')] = -38.98/180*pi
# sim_state.qpos[sim.model.get_joint_qpos_addr('front_left_ankle')] = 37.90/180*pi
# sim.set_state(sim_state)

# print(sim_state)
viewer = MjViewer(sim)

# for x in np.linspace(0,pi*2):
#     # sim_state = sim.get_state()
#     # print(sim_state.qpos)
#     # sim_state.qpos[q_index] = x
#     # sim.set_state(sim_state)
#     sim.step()
#     # sim.forward()
#     viewer.render()
#     # time.sleep(0.05)

viewer.render()
time.sleep(1)
while True:
    sim.step()
    viewer.render()
    # time.sleep(0.1)

# i = 0
# while True:
#     print(i)
#     viewer.render()
#     i += 1
