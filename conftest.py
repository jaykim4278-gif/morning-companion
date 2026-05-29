"""pytest 루트 — repo root 를 sys.path 에 추가해 `execution.*` / `src.*` import 가능."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
