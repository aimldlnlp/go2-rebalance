"""Resolve source entries in historical manifests after the directory move."""
from pathlib import Path


def project_file(root, name):
    root=Path(root)
    name=Path(name)
    original=root/name
    if original.exists() or name.parent!=Path('.') or name.suffix!='.py':
        return original
    folder='tests' if name.name.startswith('test_') else 'research'
    moved=root/folder/name
    return moved if moved.is_file() else original


def source_files(root):
    root=Path(root)
    return sorted([*root.glob('*.py'),*(root/'research').glob('*.py'),*(root/'tests').glob('*.py')])
