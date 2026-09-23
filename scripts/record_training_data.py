import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.app.runtime import main
main(["--source","vision","--backend","sim","--record-emg"])
