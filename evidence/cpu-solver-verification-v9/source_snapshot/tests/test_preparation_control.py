import numpy as np
import pytest

from sand_atlas_dem.preparation_control import OverlapControl, fast_precontact_cap


def test_hold_hysteresis_and_restart():
    control = OverlapControl()
    np.testing.assert_array_equal(control.command([-50] * 3, 0.009), [0] * 3)
    resumed = OverlapControl(**control.state())
    np.testing.assert_array_equal(resumed.command([-50] * 3, 0.007), [0] * 3)
    np.testing.assert_allclose(resumed.command([-50] * 3, 0.006), [-2.5] * 3)
    assert not resumed.held
    np.testing.assert_array_equal(resumed.command([-50] * 3, 0.008), [0] * 3)


def test_taper_and_pressure_relief():
    control = OverlapControl()
    np.testing.assert_allclose(control.command([-50, 2, -1], 0.007), [-1.25, 2, -0.25])
    np.testing.assert_array_equal(control.command([-50, 2, -1], 0.009), [0, 2, 0])


def test_rising_overlap_does_not_resume():
    control = OverlapControl(held=True, previous_overlap=0.005)
    np.testing.assert_array_equal(control.command([-50] * 3, 0.0055), [0] * 3)


@pytest.mark.parametrize("overlap", [float("nan"), float("inf"), -0.001])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        OverlapControl().command([-5] * 3, overlap)


def test_thresholds_preserve_guard():
    with pytest.raises(ValueError):
        OverlapControl(hold_at=0.01)


def test_precontact_affine_clearance_and_strain_bound():
    dt = 125 * 2e-9
    log_gap = 0.02
    cap = fast_precontact_cap(1000, 4000, dt, log_gap)
    assert cap == pytest.approx(4000)
    assert cap * dt <= 0.001 * (1 + 1e-15)
    assert np.exp(log_gap - cap * dt) > 1
    assert fast_precontact_cap(1000, 4000, dt, -0.1) == 1000
    assert fast_precontact_cap(1000, 4000, dt, 0.0015) == pytest.approx(1500)
