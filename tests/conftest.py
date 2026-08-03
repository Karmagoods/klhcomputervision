"""Pytest configuration for klhcomputervision test suite.

Adds the project root to sys.path so that ``services.*`` modules can be
imported without installing the package.  Also prevents pytest from trying
to collect Streamlit app files as test modules.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

collect_ignore_glob = ["../app.py", "../computervision.py", "../pages/**/*.py"]
