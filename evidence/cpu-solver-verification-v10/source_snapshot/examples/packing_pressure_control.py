"""Pressure feedback for exploratory packing studies; positive rates expand."""
import numpy as np


def pressure_rates(stress, overlap, protocol, batch_time):
    mode = protocol.get('pressure_control', 'mean')
    stress = np.asarray(stress, dtype=float)
    target = protocol['target_pressure_Pa']
    if mode not in {'mean', 'diagonal'}:
        raise ValueError('Unknown pressure-control mode')
    if stress.shape != (3, 3) or not np.isfinite(stress).all():
        raise ValueError('Finite 3x3 stress required')
    if target <= 0 or not np.isfinite(target) or batch_time <= 0 or not np.isfinite(batch_time):
        raise ValueError('Positive finite pressure and batch duration required')
    taper_start, hold = protocol['taper_overlap'], protocol['hold_overlap']
    if not 0 <= taper_start < hold or overlap < 0 or not np.isfinite(overlap):
        raise ValueError('Invalid overlap controller')
    diagonal = np.diag(stress)
    measured = diagonal if mode == 'diagonal' else np.full(3, diagonal.mean())
    requested = np.clip(protocol['servo_gain']*(measured/target-1), -protocol['max_rate_s'], protocol['max_rate_s'])
    taper = float(np.clip((hold-overlap)/(hold-taper_start), 0., 1.))
    rates = np.maximum(requested, 0)+np.minimum(requested, 0)*taper
    limit = protocol['max_batch_log_strain']/batch_time
    return np.clip(rates, -limit, limit), taper
