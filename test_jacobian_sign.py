import math
from mujoco_controller import Mat_JRM, ik, getPhi

L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0
target_L0 = 0.2
target_phi0 = math.pi / 2
p1, p4 = ik(target_L0, target_phi0, L1, L2, L3, L4, L5)
p2, p3, L0, phi0 = getPhi(p1, p4, L1, L2, L3, L4, L5)

J = Mat_JRM(phi0, p1, p2, p3, p4, L0, L1, L4)
print("J =", J)
print("tau for F0=1 (push down):", J @ [1, 0])
