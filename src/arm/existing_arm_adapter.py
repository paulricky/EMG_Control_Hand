from robot_controller import JointCommand

class ExistingArmAdapter:
    """Keeps shoulder/base/elbow neutral; sends only configured wrist joints."""
    def __init__(self,controller,neutral,target_joints=("wrist_pitch","wrist_roll","wrist_yaw")):
        self.controller=controller; self.neutral=dict(neutral); self.target_joints=tuple(target_joints)
        allowed={"wrist_flex","wrist_yaw","wrist_roll","wrist_pitch"}
        if len(self.target_joints)!=3 or not set(self.target_joints)<=allowed: raise ValueError("invalid wrist target joints")
    def connect(self): self.controller.connect(); return self
    def send_wrist(self,wrist_rad):
        command=dict(self.neutral)
        command.update(dict(zip(self.target_joints,map(float,wrist_rad))))
        return self.controller.send_if_due(JointCommand(gripper_open01=1.0,**command))
    def close(self): self.controller.disconnect()
