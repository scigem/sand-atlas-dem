"""Draft structural/semantic contract for ordered incremental recorded paths.

Validation does not integrate a model or check initial-target consistency.
Finite coefficient rows act on orthonormal Mandel components xx,yy,zz,xy,yz,xz.
Targets are absolute projections; strain targets are dimensionless, stress Pa.
The coordinate orders loading and is not physical time. Input is never sorted,
rescaled, deduplicated, or otherwise mutated. C++ must repeat validation.
"""
import json
from pathlib import Path
from atlas.dsl import ValidationError
from atlas.structure import check

VERSION = 'experimental-recorded-path-0.1'
RANK_PIVOT_TOLERANCE = 1e-14
SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'schemas/recorded-path.schema.json'


def _full_rank(rows):
    # Row-max scaling prevents overflow/underflow for valid finite coefficients.
    # This scaling is private to rank diagnosis, never the recorded definition.
    a = [[v/max(abs(x) for x in row) for v in row] for row in rows]
    for k in range(6):
        pivot = max(range(k,6),key=lambda i:abs(a[i][k]))
        if abs(a[pivot][k]) < RANK_PIVOT_TOLERANCE:
            return False
        a[k],a[pivot] = a[pivot],a[k]
        for i in range(k+1,6):
            factor = a[i][k]/a[k][k]
            for j in range(k+1,6):
                a[i][j] -= factor*a[k][j]
            a[i][k] = 0.
    return True


def validate_recorded_path(spec):
    """Validate and return the original specification without changing it.

    Pure-strain coefficient matrices must have numerical rank six under a
    row-scaled partial-pivot test. Mixed controls cannot be ranked using their
    coefficients alone: the constitutive stiffness enters the control Jacobian.
    """
    try:
        check(spec,json.loads(SCHEMA_PATH.read_text()))
    except OverflowError as exc:
        raise ValidationError('numeric input exceeds finite floating-point range') from exc
    if not spec['coordinate']['name'].strip():
        raise ValidationError('coordinate name must contain non-whitespace text')
    for i,row in enumerate(spec['controls']):
        if not any(v != 0 for v in row['coefficients']):
            raise ValidationError(f'controls[{i}]: coefficient row must be nonzero')
    if all(row['kind']=='strain' for row in spec['controls']):
        if not _full_rank([row['coefficients'] for row in spec['controls']]):
            raise ValidationError('all-strain coefficient matrix is numerically rank deficient')
    previous = spec['knots'][0]['coordinate']
    for i,knot in enumerate(spec['knots'][1:],1):
        if not knot['coordinate'] > previous:
            raise ValidationError(f'knots[{i}]: coordinates must be strictly increasing in input order')
        previous = knot['coordinate']
    return spec


MATERIAL_DEFAULTS = dict(residual_tolerance=2e-10, rate_tolerance=1e-6,
                         max_iterations=45, max_depth=14, max_substeps=20000,
                         max_backtracks=11)
CONTROL_DEFAULTS = dict(control_tolerance=1e-10, max_iterations=30,
                        max_backtracks=10, max_material_trials=20000,
                        finite_difference_step=1e-7)


RESTART_DEFAULTS = dict(max_restarts=1, restart_step=1e-4, restart_levels=8)

def validate_recorded_run(spec):
    """Validate a closed request; model compatibility is checked by the engine."""
    try:
        check(spec, json.loads(SCHEMA_PATH.with_name('recorded-run.schema.json').read_text()))
    except OverflowError as exc:
        raise ValidationError('numeric input exceeds finite floating-point range') from exc
    validate_recorded_path(spec['path'])
    def exact_integers(value):
        if type(value) is int and float(value) != value:
            raise ValidationError('integer input is not exactly representable in binary64')
        if isinstance(value, dict):
            for item in value.values(): exact_integers(item)
        elif isinstance(value, list):
            for item in value: exact_integers(item)
    exact_integers(spec)
    for field in ('parameters', 'initial'):
        if field in spec and not spec[field]:
            raise ValidationError(f'{field}: omit to use defaults; explicit arrays must be nonempty')
    return spec


def recorded_wire(spec):
    """Lossless ordered numeric transport; no sorting, interpolation or fitting."""
    validate_recorded_run(spec)
    path = spec['path']
    controls = [v for row in path['controls']
                for v in [int(row['kind'] == 'stress'), *row['coefficients']]]
    knots = [v for knot in path['knots'] for v in [knot['coordinate'], *knot['targets']]]
    return controls, knots


def execute_recorded(spec, solver, timeout=60):
    """Execute validated JSON through the native CLI and preserve its failure prefix.

    Temporary wire files avoid OS argument-size limits at the 5001-knot bound.
    Does not compile, fit parameters, prepare a specimen or alter observations.
    """
    import hashlib
    import subprocess
    import tempfile

    controls, knots = recorded_wire(spec)
    material = MATERIAL_DEFAULTS | spec.get('integration_settings', {})
    control = CONTROL_DEFAULTS | spec.get('control_settings', {})
    restarts = RESTART_DEFAULTS | spec['restart_settings'] if 'restart_settings' in spec else None
    csv = lambda values: ','.join(repr(float(v)) for v in values)
    with tempfile.TemporaryDirectory(prefix='atlas-recorded-') as directory:
        c, k = Path(directory)/'controls.csv', Path(directory)/'knots.csv'
        c.write_text(csv(controls)); k.write_text(csv(knots))
        command = [str(Path(solver).resolve()), '--mode', 'recorded', '--model', spec['model'],
                   '--initial-strain', csv(spec['path']['initial_strain']),
                   '--recorded-controls-file', str(c), '--recorded-knots-file', str(k),
                   '--integration-settings', csv(material.values()),
                   '--recorded-settings', csv(control.values())]
        if restarts is not None: command.extend(['--recorded-restarts', csv(restarts.values())])
        for field in ('parameters', 'initial'):
            if field in spec: command.extend(['--'+field, csv(spec[field])])
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    try:
        result = json.loads(process.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('recorded solver did not return JSON') from exc
    if not isinstance(result, dict) or type(result.get('ok')) is not bool:
        raise RuntimeError('recorded solver returned an invalid status')
    if process.returncode != (0 if result['ok'] else 1):
        raise RuntimeError('recorded solver exit code disagrees with result')
    if result['ok'] or 'api_version' in result:
        if (result.get('api_version') != 'experimental-recorded-replay-0.1'
                or result.get('model') != spec['model']
                or result.get('controls_wire') != controls or result.get('knots_wire') != knots
                or result.get('initial_strain') != spec['path']['initial_strain']
                or result.get('control_settings') != control
                or any(result.get('integration_settings', {}).get(key) != value
                       for key, value in material.items())):
            raise RuntimeError('recorded solver response does not match request or settings')
    if result['ok'] or 'api_version' in result:
        if (restarts is not None and result.get('restart_settings') != restarts) or (restarts is None and any(key in result for key in ('restart_settings','restart_diagnostics'))):
            raise RuntimeError('recorded solver restart policy does not match request')
        if restarts is not None:
            diagnostics=result.get('restart_diagnostics', {})
            if set(diagnostics) != {'searches','material_trials','accepted'} or any(type(v) is not int or v<0 for v in diagnostics.values()):
                raise RuntimeError('invalid recorded restart diagnostics')
            total=result.get('material_trials')
            if type(total) is not int or total<0:
                raise RuntimeError('invalid recorded material trial total')
            if (not diagnostics['accepted']<=diagnostics['searches'] or diagnostics['material_trials']>total
                    or (restarts['max_restarts']==0 and any(diagnostics.values()))):
                raise RuntimeError('inconsistent recorded restart diagnostics')
    result['specification'] = spec
    canonical = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
    result['request_sha256'] = hashlib.sha256(canonical.encode()).hexdigest()
    return result
