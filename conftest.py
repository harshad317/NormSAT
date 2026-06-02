"""Makes the project root importable so `pytest` finds the `rancsat` package
without needing `pip install -e .` or setting PYTHONPATH."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
