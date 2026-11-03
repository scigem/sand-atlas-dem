"""The prototype must be explicit and cannot silently enable unqualified physics."""
from dataclasses import replace

import pytest

from sand_atlas_dem.spec import Experiment


def test_source_normal_default_and_serialization():
    assert Experiment().contact_normal == 'target-gradient'
    spec = Experiment(contact_geometry='mesh', friction=0,
                      contact_normal='source-surface-experimental')
    assert Experiment(**spec.to_dict()) == spec


@pytest.mark.parametrize('change', [dict(contact_geometry='sdf'),dict(friction=.1),
                                    dict(normal_viscosity=1.),dict(contact_normal='source')])
def test_source_normal_rejects_unsupported_settings(change):
    spec = Experiment(contact_geometry='mesh', friction=0,
                      contact_normal='source-surface-experimental')
    with pytest.raises(ValueError):
        replace(spec, **change)
