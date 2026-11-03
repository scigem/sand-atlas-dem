"""Checked scalar elementary operations with stable binary64 primitives."""
import math
LOG1P_VERSION = 'scalar-log1p-libm-checked-v1'

def log1p(value):
    if type(value) not in (int, float):
        raise ValueError('log1p requires a finite real scalar')
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('log1p requires a finite real scalar') from exc
    if not math.isfinite(value) or value <= -1:
        raise ValueError('log1p requires finite x > -1')
    result = math.log1p(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite log1p output')
    return result
