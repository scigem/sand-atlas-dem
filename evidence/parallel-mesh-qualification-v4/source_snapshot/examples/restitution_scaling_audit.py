"""Independent dimensional/scalar audit, never a DEM dynamics fallback.

Ordinary spring-dashpot restitution is contrasted with area-weighted traction.
The examined alternatives are analytical examples, not benchmark input recipes.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from sand_atlas_dem.util import read_json, sha256, write_json


def damping_ratio(restitution):
    if not np.isfinite(restitution) or not 0 < restitution <= 1:
        raise ValueError("Linear underdamped restitution requires 0 < e <= 1")
    logarithm = -np.log(restitution)
    return float(logarithm / np.sqrt(np.pi**2 + logarithm**2))


def linear_restitution(ratio):
    if not np.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("Underdamped ratio requires 0 <= zeta < 1")
    return float(np.exp(-np.pi * ratio / np.sqrt(1 - ratio**2)))


def patch_coefficients(nodes, area, stiffness, mass, restitution):
    if (
        type(nodes) is not int
        or nodes < 1
        or not np.isfinite([area, stiffness, mass]).all()
        or min(area, stiffness, mass) <= 0
    ):
        raise ValueError("Positive patch quantities and integer node count required")
    ratio = damping_ratio(restitution)
    node_area = area / nodes
    # k_i has units N/m, obtained from traction stiffness times tributary area.
    node_stiffness = stiffness * node_area
    full_mass = nodes * 2 * ratio * np.sqrt(mass * node_stiffness)
    area_partition = nodes * 2 * ratio * np.sqrt((mass / nodes) * node_stiffness)
    return dict(
        full_body_mass_at_every_node=float(full_mass),
        area_partitioned_mass=float(area_partition),
    )


def sphere_cap(radius, depth, stiffness, viscosity, compression_speed):
    if not np.isfinite(
        [radius, depth, stiffness, viscosity, compression_speed]
    ).all() or not (
        radius > 0
        and 0 < depth <= 2 * radius
        and stiffness > 0
        and viscosity >= 0
        and compression_speed >= 0
    ):
        raise ValueError(
            "Finite positive cap geometry and compressive loading required"
        )
    area = 2 * np.pi * radius * depth
    return dict(
        elastic_force=np.pi * stiffness * radius * depth**2,
        damping_force=viscosity * area * compression_speed,
        dissipation_power=viscosity * area * compression_speed**2,
        tangent_stiffness=2 * np.pi * stiffness * radius * depth,
        damping_coefficient=viscosity * area,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.policy)
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), args.policy]
    write_json(args.output / "source-freeze.json", {str(p): sha256(p) for p in files})
    rows = []
    for restitution in policy["restitution_values"]:
        ratio = damping_ratio(restitution)
        mass, stiffness = policy["mass_kg"], policy["ordinary_stiffness_n_per_m"]
        coefficient = 2 * ratio * np.sqrt(mass * stiffness)
        contact_time = np.pi * np.sqrt(mass / stiffness) / np.sqrt(1 - ratio**2)
        solution = solve_ivp(
            lambda t, y: [y[1], -(stiffness * y[0] + coefficient * y[1]) / mass],
            [0, contact_time],
            [0, 1],
            rtol=2e-12,
            atol=2e-13,
        )
        rows.append(
            dict(
                restitution=restitution,
                ratio=ratio,
                coefficient_kg_per_s=coefficient,
                return_displacement=float(solution.y[0, -1]),
                measured_restitution=float(-solution.y[1, -1]),
                restitution_error=abs(float(-solution.y[1, -1]) - restitution),
            )
        )
    patch = [
        dict(
            nodes=n,
            **patch_coefficients(
                n,
                policy["patch_area_m2"],
                policy["traction_stiffness_n_per_m3"],
                policy["mass_kg"],
                policy["patch_restitution"],
            ),
        )
        for n in policy["node_counts"]
    ]
    baseline = patch[0]
    for row in patch:
        row["full_mass_growth"] = (
            row["full_body_mass_at_every_node"]
            / baseline["full_body_mass_at_every_node"]
        )
        row["area_mass_growth"] = (
            row["area_partitioned_mass"] / baseline["area_partitioned_mass"]
        )
    ratio = damping_ratio(policy["patch_restitution"])
    fractions = [
        dict(
            contact_area_fraction=f,
            effective_ratio=ratio * np.sqrt(f),
            effective_linear_restitution=linear_restitution(ratio * np.sqrt(f)),
        )
        for f in policy["contact_area_fractions"]
    ]
    cap = []
    for depth in policy["cap_depths_m"]:
        result = sphere_cap(
            policy["cap_radius_m"],
            depth,
            policy["traction_stiffness_n_per_m3"],
            policy["normal_viscosity_n_s_per_m3"],
            policy["compression_speed_m_per_s"],
        )
        result["depth_m"] = depth
        result["local_tangent_ratio"] = result["damping_coefficient"] / (
            2 * np.sqrt(policy["mass_kg"] * result["tangent_stiffness"])
        )
        cap.append(result)
    gates = dict(
        oscillator=max(r["restitution_error"] for r in rows)
        < policy["maximum_oscillator_error"],
        divergent_full_mass=np.allclose(
            [r["full_mass_growth"] for r in patch],
            np.sqrt(np.array(policy["node_counts"]) / policy["node_counts"][0]),
            rtol=policy["scaling_relative_tolerance"],
            atol=0,
        ),
        invariant_area_mass=np.allclose(
            [r["area_mass_growth"] for r in patch],
            1,
            rtol=policy["scaling_relative_tolerance"],
            atol=0,
        ),
        no_global_restitution=fractions[0]["effective_linear_restitution"]
        != fractions[-1]["effective_linear_restitution"],
        cap_ratio=np.allclose(
            np.array([r["local_tangent_ratio"] for r in cap])
            / cap[0]["local_tangent_ratio"],
            np.sqrt(np.array(policy["cap_depths_m"]) / policy["cap_depths_m"][0]),
            rtol=policy["scaling_relative_tolerance"],
            atol=0,
        ),
    )
    write_json(
        args.output / "report.json",
        dict(
            identity=policy["identity"],
            passed=bool(all(gates.values())),
            gates=gates,
            oscillators=rows,
            patch_scaling=patch,
            contact_fraction_effect=fractions,
            sphere_cap=cap,
            source_derived_benchmark_damping=False,
            gpu_dynamics_executed=False,
            scientific_admission=False,
        ),
    )
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
