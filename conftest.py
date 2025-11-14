# conftest.py en la raíz del repo MLops_team41
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ruta a MCE/Model_construction_and_evaluation/src
mce_root = ROOT / "MCE" / "Model_construction_and_evaluation"
src_root = mce_root / "src"

for p in (mce_root, src_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))