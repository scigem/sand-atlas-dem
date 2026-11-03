"""Explicit source-unit conversions for browser assets; no quantity/sign inference."""
from copy import deepcopy
import math

# SI target, scale, additive offset. Aliases are exact, reviewed source labels.
UNITS = {}
for aliases, target, scale, offset in [
    (['Pa'], 'Pa', 1, 0), (['kPa', '(kPa)', '[kPa]', "s' [kPa]", 'kN/m2'], 'Pa', 1000, 0),
    (['MPa'], 'Pa', 1e6, 0), (['%', '(%)', '[%]', 'in %'], '1', .01, 0),
    (['1', '-', '(-)', '[-]', 'unitless', 'mm/mm'], '1', 1, 0),
    (['mm', '(mm)', 'in mm'], 'm', .001, 0),
    (['µm'], 'm', 1e-6, 0), (['µm/s'], 'm/s', 1e-6, 0),
    (['mm^2'], 'm²', 1e-6, 0), (['(mm3)'], 'm³', 1e-9, 0),
    (['cm^3', 'ml'], 'm³', 1e-6, 0), (['N'], 'N', 1, 0),
    (['ml/2 g'], 'm³/kg', .0005, 0),
    (['L'], 'm³', .001, 0), (['10^-16 m^2'], 'm²', 1e-16, 0),
    (['N・m', 'Nm'], 'N·m', 1, 0),
    (['kN'], 'N', 1000, 0), (['lbf'], 'N', 4.4482216152605, 0),
    (['s', 'Sec', '(sec)', 'sec'], 's', 1, 0), (['ms'], 's', .001, 0),
    (['min'], 's', 60, 0), (['h', 'hour'], 's', 3600, 0),
    (['dny'], 's', 86400, 0),
    (['%/h'], 's⁻¹', .01/3600, 0), (['degree', 'deg'], 'rad', math.pi/180, 0),
    (['\\circC', '°C'], 'K', 1, 273.15), (['kg/m3'], 'kg/m³', 1, 0),
    (['g/cm^3'], 'kg/m³', 1000, 0),
    (['W/m·K'], 'W/(m·K)', 1, 0),
    (['J/kg·K'], 'J/(kg·K)', 1, 0), (['J/m3·K'], 'J/(m³·K)', 1, 0),
    (['m/s', 'kT [m/s]', 'k10 [m/s]'], 'm/s', 1, 0),
    (['V'], 'V', 1, 0), (['cycles', '回'], 'cycles', 1, 0),
]:
    for alias in aliases:
        UNITS[alias] = (target, scale, offset)


def sanitize_units(record):
    """Return a converted copy, retaining every row, source unit and transform.

    Unknown units (including logarithms with a source reference pressure) remain
    unchanged and explicitly unresolved. Unit conversion never establishes that
    an apparatus signal is specimen stress, effective stress or strain.
    """
    if record['kind'] == 'experiment':
        return record  # Validated SI/compression-positive import contract.
    result = deepcopy(record)
    for column in result['columns']:
        source = column['unit']
        conversion = UNITS.get(source)
        column['source_unit'] = source
        column['unit_status'] = 'standardized' if conversion else 'unresolved'
        if not conversion:
            continue
        target, scale, offset = conversion
        column.update(unit=target, unit_conversion=dict(scale=scale, offset=offset))
        for row in result['rows']:
            if column['key'] in row:
                value = row[column['key']] * scale + offset
                if not math.isfinite(value):
                    raise ValueError('Unit conversion produced a nonfinite measurement')
                row[column['key']] = value
    return result


def comparison_metadata(record):
    """Only reviewed physical quantities can be offered for model overlays."""
    if record['kind'] != 'experiment':
        return dict(plots=[], loading='unreviewed')
    pairs = dict(stress=('axial_strain', 'q'), path=('p', 'q'),
                 volume=('axial_strain', 'volumetric_strain'),
                 pore=('axial_strain', 'pore_pressure'))
    plots = [name for name, keys in pairs.items()
             if any(all(key in row for key in keys) for row in record['rows'])]
    strain = [row['axial_strain'] for row in record['rows']]
    # Ignore acquisition jitter; require a sustained excursion in both directions.
    tolerance = max(1e-8, (max(strain)-min(strain))*.05)
    low = high = strain[0]
    rising = falling = False
    for value in strain:
        rising |= value-low > tolerance
        falling |= high-value > tolerance
        low, high = min(low, value), max(high, value)
    return dict(plots=plots, loading='reversing' if rising and falling else 'monotonic')
