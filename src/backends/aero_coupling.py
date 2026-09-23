"""Official SDK joint-to-actuation equations and Menagerie control conversion.

Coefficients mirror Chestnut Robotics' joints_to_actuations.py (angles in
degrees at the public boundary). Menagerie tendon position controls are metres.
"""
import numpy as np

SDK_ACTUATOR_NAMES = ("thumb_cmc_abd_act", "thumb_cmc_flex_act", "thumb_tendon_act",
                      "index_tendon_act", "middle_tendon_act", "ring_tendon_act", "pinky_tendon_act")
MUJOCO_ACTUATOR_NAMES = ("right_index_A_tendon", "right_middle_A_tendon", "right_ring_A_tendon",
                         "right_pinky_A_tendon", "right_thumb_A_cmc_abd", "right_th1_A_tendon", "right_th2_A_tendon")
MUJOCO_CTRL_LOW = np.array([.058520]*4 + [-.1, .026152, .081568])
MUJOCO_CTRL_HIGH = np.array([.110387]*4 + [1.75, .038389, .112138])


def joints_to_sdk_actuations_deg(joints_deg):
    q=np.deg2rad(np.asarray(joints_deg,float).reshape(16)); result=[]
    abd, flex, mcp, ip=q[:4]
    result.extend((abd, (2.5*abd+12.4931*flex)/9.0,
                   (2.5*abd-2.5*flex+9.4372*mcp+12.5*ip)/9.0))
    for s in (4,7,10,13): result.append((12.4912*q[s]+7.3211*q[s+1]+9*q[s+2])/9.0)
    return np.rad2deg(result)


def joints_to_mujoco_ctrl(joints_deg):
    a=np.deg2rad(joints_to_sdk_actuations_deg(joints_deg))
    # Positive motor rotation reels in a 9 mm pulley, shortening the tendon.
    controls=np.array([*(.110387 - .009*a[3:7]), a[0], .038389-.009*a[1], .112138-.009*a[2]])
    return np.clip(controls, MUJOCO_CTRL_LOW, MUJOCO_CTRL_HIGH)
