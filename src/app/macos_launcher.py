"""Transparent launcher for MuJoCo's macOS Cocoa event loop."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MJPYTHON_MARKER = "AERO_RUNNING_UNDER_MJPYTHON"


def find_mjpython() -> str | None:
    """Prefer the mjpython installed beside the active Python interpreter."""
    candidates = (Path(sys.executable).with_name("mjpython"), Path(sys.prefix) / "bin" / "mjpython")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("mjpython")


def should_relaunch(show_mujoco: bool, platform: str | None = None, environ=None) -> bool:
    env = os.environ if environ is None else environ
    return (platform or sys.platform) == "darwin" and show_mujoco and env.get(MJPYTHON_MARKER) != "1"


def ensure_mjpython(show_mujoco: bool, argv=None, module: str | None = None) -> bool:
    """Replace this process with mjpython when the passive macOS viewer needs it."""
    if not should_relaunch(show_mujoco):
        return False
    executable = find_mjpython()
    if executable is None:
        expected = Path(sys.executable).with_name("mjpython")
        raise RuntimeError(
            f"MuJoCo's macOS viewer requires mjpython, but it was not found in the active "
            f"environment {sys.prefix!r}. Expected an executable at {expected}."
        )
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = [executable, "-m", module, *arguments] if module else [executable, os.path.abspath(sys.argv[0]), *arguments]
    env = os.environ.copy()
    env[MJPYTHON_MARKER] = "1"
    os.execve(executable, command, env)
    return True  # pragma: no cover - execve does not return
