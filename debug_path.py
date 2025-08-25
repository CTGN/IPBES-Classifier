#!/usr/bin/env python3
from pathlib import Path
import sys

current = Path(__file__).absolute()
print(f"Current file: {current}")

project_root = current
for i in range(5):
    project_root = project_root.parent
    print(f"Parent {i+1}: {project_root}")
    if (project_root / 'pyproject.toml').exists():
        print(f"Found project root at level {i+1}: {project_root}")
        print(f"Adding to sys.path: {project_root}")
        sys.path.insert(0, str(project_root))
        break

print(f"sys.path[0]: {sys.path[0]}")

try:
    from src.config import CONFIG
    print("SUCCESS: src.config imported!")
    print(f"CONFIG keys: {len(CONFIG)}")
except Exception as e:
    print(f"FAILED: {e}")

