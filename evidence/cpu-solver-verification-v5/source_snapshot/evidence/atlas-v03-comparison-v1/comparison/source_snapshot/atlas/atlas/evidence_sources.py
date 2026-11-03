"""Explicit cache-free source selection for NEW prospective evidence closures.

Historical freezes are immutable. Never regenerate their source lists with this
helper. Callers choose roots and suffixes and separately pin all chosen bytes.
"""
from pathlib import Path

def source_allowed(relative,suffixes):
    path=Path(relative)
    if path.is_absolute()or '..'in path.parts:
        raise ValueError('source selection requires a relative path within its root')
    return path.suffix not in {'.pyc','.pyo'} and path.suffix in suffixes and not any(part=='__pycache__'or part.startswith('.')for part in path.parts)

def selected_files(base,suffixes):
    base=Path(base)
    if not base.is_dir():raise ValueError('source selection requires an existing directory')
    result=[]
    for path in base.rglob('*'):
        if source_allowed(path.relative_to(base),suffixes)and path.is_file():
            if path.is_symlink():raise ValueError('selected evidence source must not be a symlink: '+str(path))
            result.append(path)
    return tuple(sorted(result))
