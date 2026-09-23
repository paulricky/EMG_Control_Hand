import sys
import os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from src.app.macos_launcher import ensure_mjpython
from src.app.runtime import main

if __name__ == "__main__":
    ensure_mjpython(True)
    main([
        "--source", "vision",
        "--backend", "sim",
        "--show-camera",
        "--show-mujoco",
        *sys.argv[1:],
    ])
