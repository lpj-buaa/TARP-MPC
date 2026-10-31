from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
# Test both existing Python3.11 runtimes without modifying either environment.
# GPU runtime owns torch/sklearn; physical-only packages are read-only fallbacks.
physical_packages=Path(__file__).resolve().parents[2]/'tarp_risk_discovery/.venv/Lib/site-packages'
if str(physical_packages) not in sys.path: sys.path.append(str(physical_packages))
