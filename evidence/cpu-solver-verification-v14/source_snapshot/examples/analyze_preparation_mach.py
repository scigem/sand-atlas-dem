"""Read-only dimensional/contact-speed diagnostics; not a measured wave speed."""

import argparse
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from sand_atlas_dem.simulation import initial_assembly
from sand_atlas_dem.spec import Experiment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--replay", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(exist_ok=False)
(out / "analyze_preparation_mach.py").write_bytes(Path(__file__).read_bytes())
paths = [
    root / "runs/hostun-balanced424-resume-v1/study/run" / n / "checkpoint.h5"
    for n in ["chunk-0004", "chunk-0005"]
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


replay_path = out / "observed-replay.json"
replay_path.write_bytes(args.replay.read_bytes())
protocol = dict(
    identity=out.name,
    driver_sha256=sha(__file__),
    replay_sha256=sha(replay_path),
    assembly_source_sha256=sha(root / "src/sand_atlas_dem/simulation.py"),
    checkpoints={str(p): sha(p) for p in paths},
    rates_s_inv=[50, 5, 0],
    formulas=dict(
        nominal_c="sqrt(kn_star*d/rho), equivalent E=kn_star*d: dimensional scale only",
        pair_k="sum(0.5*kn_star*A) over both directed active surface passes",
        pair_c="mean(d_i,d_j)*sqrt(k_pair/reduced_mass): isolated normal-contact chain-speed proxy, not bulk c",
        contact_projection="kn_pair along selected deepest-contact normal=sum(0.5*kn_star*A*(n dot n_deepest)^2)",
        damping_ratio="drag/(2*omega_pair)",
        overlap_loading_ratio="drag*abs(n_deepest dot (rate*branch))/(omega_projected^2*0.01*min(d_i,d_j)); frozen-normal translation-only forced-damped pair estimate",
        inertial_scale="rate*median(d)*sqrt(rho/p); isotropic compression rate substituted for shear rate, not a constitutive I",
    ),
    scope="No uniform bulk wave speed is assumed for a disconnected dilute contact network. No new numerical gate or safety claim. Frozen normal stiffness neglects rotation, prestress, curvature, sliding and tangential coupling. CPU analysis only; Metal replays remain serial.",
)
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
shapefile = None
rows = []
for path in paths:
    with h5py.File(path) as f:
        spec = Experiment(**json.loads(f.attrs["experiment"]))
        attrs = dict(f.attrs)
        state = {k: f[k][:] for k in f}
        last = json.loads(f.attrs["rows"])[-1]
    if shapefile is None:
        shapefile = Path(spec.shape_library)
        protocol["shape_sha256"] = sha(shapefile)
        assert protocol["shape_sha256"] == attrs["shape_sha256"]
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        with h5py.File(shapefile) as f:
            shapes = [
                SimpleNamespace(
                    volume=float(f[k].attrs["volume"]),
                    radius=float(f[k].attrs["radius"]),
                    id=f[k].attrs["id"],
                )
                for k in sorted(f, key=int)
            ]
    library = SimpleNamespace(shapes=shapes)
    _, _, ids, scales = initial_assembly(library, spec)
    diam = np.array([(6 * s.volume / np.pi) ** (1 / 3) for s in shapes])
    length = float(np.median(diam))
    tunit = np.sqrt(spec.density * length / spec.normal_stiffness)
    d = diam[ids] * scales
    mass = np.array([s.volume for s in shapes])[ids] * spec.density * scales**3
    vel = (
        (state["velocity"][:, :3].astype(float) + state["velocityRemainder"][:, :3])
        * length
        / tunit
    )
    pos = (
        state["position"][:, :3].astype(float) + state["positionRemainder"][:, :3]
    ) * length
    cell = state["cell"][0, :3].astype(float) * length
    keys = state["meta"]
    normals = state["normalArea"][:, :3].astype(float)
    areas = state["normalArea"][:, 3].astype(float) * length**2
    stiffness = 0.5 * spec.normal_stiffness * areas
    pairs = np.sort(keys[:, :2], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    kp = np.bincount(inverse, weights=stiffness)
    mred = (
        mass[unique[:, 0]]
        * mass[unique[:, 1]]
        / (mass[unique[:, 0]] + mass[unique[:, 1]])
    )
    omega = np.sqrt(kp / mred)
    dmean = d[unique].mean(axis=1)
    cp = dmean * omega
    # Recover deepest source sample and normal directly from the retained checkpoint.
    # Its identity was independently observed in the Metal replay, not inferred from force.
    replay = json.loads(replay_path.read_text())
    label = "earlier" if attrs["step"] == 2250000 else "near-limit"
    case = next(c for c in replay["cases"] if c["name"] == label + "-original")
    worst = case["rows"][0]["worst_contact"]
    i = worst["source_body"]
    j = worst["target_body"]
    node = worst["source_node"]
    contact = np.flatnonzero(
        (keys[:, 0] == i) & (keys[:, 1] == j) & (keys[:, 2] == node)
    )[0]
    n = normals[contact]
    n /= np.linalg.norm(n)
    pairmask = (pairs[:, 0] == min(i, j)) & (pairs[:, 1] == max(i, j))
    kn = float(np.sum(stiffness[pairmask] * (normals[pairmask] @ n) ** 2))
    mu = mass[i] * mass[j] / (mass[i] + mass[j])
    wn = np.sqrt(kn / mu)
    branch = pos[j] - pos[i]
    branch -= np.round(branch / cell) * cell
    pairindex = int(
        np.flatnonzero((unique[:, 0] == min(i, j)) & (unique[:, 1] == max(i, j)))[0]
    )
    parents = np.arange(spec.particles)

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for graph_i, graph_j in unique:
        parents[find(int(graph_i))] = find(int(graph_j))
    counts = np.bincount([find(i) for i in range(spec.particles)])
    data = dict(
        largest_contact_component_bodies=int(counts.max()),
        step=int(attrs["step"]),
        median_diameter_m=float(np.median(d)),
        cell_m=cell.tolist(),
        nominal_c_m_s=[
            float(np.min(np.sqrt(spec.normal_stiffness * d / spec.density))),
            float(np.max(np.sqrt(spec.normal_stiffness * d / spec.density))),
        ],
        peculiar_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
        peculiar_speed_max_m_s=float(np.linalg.norm(vel, axis=1).max()),
        active_pairs=len(unique),
        active_bodies=len(np.unique(unique)),
        pair_speed_proxy_m_s=dict(
            min=float(cp.min()), median=float(np.median(cp)), max=float(cp.max())
        ),
        damping_ratio_pair=dict(
            min=float((1e5 / (2 * omega)).min()),
            median=float(np.median(1e5 / (2 * omega))),
            max=float((1e5 / (2 * omega)).max()),
        ),
        deepest_pair=dict(
            source_body=i,
            target_body=j,
            source_shape=shapes[int(ids[i])].id,
            target_shape=shapes[int(ids[j])].id,
            normal_stiffness_N_m=kn,
            reduced_mass_kg=float(mu),
            omega_normal_rad_s=float(wn),
            normal_speed_proxy_m_s=float(np.mean([d[i], d[j]]) * wn),
            damping_ratio=float(1e5 / (2 * wn)),
        ),
        rates=[],
    )
    for rate in [50, 5, 0]:
        u = float(rate * cell.max() / 2)
        un = float(abs(n @ (-rate * branch)))
        cnom = float(np.sqrt(spec.normal_stiffness * np.median(d) / spec.density))
        data["rates"].append(
            dict(
                rate_s_inv=rate,
                axis_boundary_speed_m_s=u,
                nominal_boundary_Mach=u / cnom,
                boundary_over_deepest_pair_speed=u / (np.mean([d[i], d[j]]) * wn),
                deepest_pair_affine_normal_closure_m_s=un,
                deepest_pair_local_Mach=un / (np.mean([d[i], d[j]]) * wn),
                damped_overlap_loading_ratio=1e5
                * un
                / (wn**2 * 0.01 * min(d[i], d[j])),
                compression_inertial_scale=float(
                    rate * np.median(d) * np.sqrt(spec.density / last["p"])
                ),
            )
        )
    rows.append(data)
report = dict(
    protocol=protocol,
    rows=rows,
    input_integrity=all(sha(p) == h for p, h in protocol["checkpoints"].items())
    and sha(shapefile) == protocol["shape_sha256"]
    and sha(__file__) == protocol["driver_sha256"]
    and sha(replay_path) == protocol["replay_sha256"]
    and sha(root / "src/sand_atlas_dem/simulation.py")
    == protocol["assembly_source_sha256"],
    scientific_admission=False,
)
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["rows"], indent=2))
