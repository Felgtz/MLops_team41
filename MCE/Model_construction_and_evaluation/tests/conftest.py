# MCE/Model_construction_and_evaluation/tests/conftest.py
import sys
from pathlib import Path

# Carpeta raíz del módulo MCE/Model_construction_and_evaluation
ROOT = Path(__file__).resolve().parents[1]
# Añadimos esa ruta al sys.path para que `import src...` funcione
sys.path.insert(0, str(ROOT))
