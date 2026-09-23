from __future__ import annotations
from abc import ABC, abstractmethod


class AeroBackend(ABC):
    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def send(self, command): ...
    @abstractmethod
    def read_state(self): ...
    @abstractmethod
    def shutdown(self): ...

    def reset(self): pass
    def step(self): pass
    def render(self): pass
    def handle_fault(self, action, last_safe_command=None): pass
