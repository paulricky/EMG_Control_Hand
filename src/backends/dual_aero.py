from .base import AeroBackend


class DualAeroBackend(AeroBackend):
    def __init__(self, simulation, hardware): self.simulation=simulation; self.hardware=hardware
    def connect(self): self.simulation.connect(); self.hardware.connect(); return self
    def send(self, command): self.simulation.send(command); self.hardware.send(command)
    def step(self): self.simulation.step()
    def render(self): self.simulation.render()
    def reset(self): self.simulation.reset()
    def read_state(self): return {"simulation":self.simulation.read_state(),"hardware":self.hardware.read_state()}
    def shutdown(self):
        try: self.hardware.shutdown()
        finally: self.simulation.shutdown()
    def handle_fault(self, action, last_safe_command=None):
        self.simulation.handle_fault(action,last_safe_command); self.hardware.handle_fault(action,last_safe_command)
