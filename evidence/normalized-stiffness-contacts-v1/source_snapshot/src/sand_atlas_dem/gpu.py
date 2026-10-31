"""GPU-resident Slang DEM engine; no CPU fallback or host contact loops."""

import math

import numpy as np

from .backend import Metal
from .boundaries import box_extents
from .util import finite_array

ERRORS = {
    2: "degenerate contact distance gradient",
    4: "excessive penetration",
    8: "bin capacity",
    16: "pair capacity",
    32: "contact capacity",
    64: "history hash full",
    128: "contact-list cycle",
    256: "periodic image ambiguity",
    512: "nonfinite state",
    1024: "mesh traversal stack capacity",
    2048: "nonperiodic domain escape",
    4096: "invalid boundary box",
}


DAMPING_STATE = ("normalDamping", "forceVelocity", "forceOmega")


WALL_STATE = (
    "wallPlane",
    "wallRemainder",
    "wallVelocity",
    "wallAccumulated",
    "wallRows",
)


RESIDUAL_STATE = (
    "positionRemainder",
    "velocityRemainder",
    "momentumRemainder",
    "rotationRemainder",
    "accumulatedRemainder",
    "cellLog",
)


class RejectedStep(RuntimeError):
    def __init__(self, flags):
        self.flags = int(flags)
        super().__init__(
            "GPU step rejected: "
            + ", ".join(v for k, v in ERRORS.items() if self.flags & k)
        )


def f4(x):
    a = np.asarray(x, np.float32)
    return np.pad(a, ((0, 0), (0, 4 - a.shape[1])))


class Scan:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        blocks = (n + 255) // 256
        self.totals = m.buffer(count=blocks)
        self.offsets = m.buffer(count=blocks)
        self.child = Scan(m, blocks) if blocks > 1 else None

    def encode(self, e, source, dest):
        self.m.kernel("blockScan", "scan.slang").dispatch(
            thread_count=[((self.n + 255) // 256) * 256, 1, 1],
            command_encoder=e,
            N=self.n,
            input=source,
            output=dest,
            totals=self.totals,
        )
        if self.child:
            self.child.encode(e, self.totals, self.offsets)
            self.m.kernel("addBlocks", "scan.slang").dispatch(
                thread_count=[self.n, 1, 1],
                command_encoder=e,
                N=self.n,
                output=dest,
                offsets=self.offsets,
            )


class Engine:
    def __init__(
        self, library, spec, positions, cell, shape_ids, scales, *, metal=None
    ):
        positions = finite_array(positions)
        cell = finite_array(cell, (3,), positive=True)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
            raise ValueError("At least one three-dimensional position required")
        if not spec.periodic and (np.any(positions < 0) or np.any(positions >= cell)):
            raise ValueError("Nonperiodic particle centres must lie in [0, cell)")
        if spec.boundary_box is not None:
            box_extents(
                np.array([p["offset"] for p in spec.planes]), spec.boundary_box, cell
            )
        self.m = metal or Metal()
        self.spec = spec
        self.library = library
        self.n = len(positions)
        self.wall_count = len(spec.planes)
        self.viscous = spec.viscous
        self.step = 0
        self.history = False
        self.active = 0
        self.capacity_retries = []
        self.length_unit = float(
            np.median([(6 * s.volume / np.pi) ** (1 / 3) for s in library.shapes])
        )
        self.time_unit = math.sqrt(
            spec.density * self.length_unit / spec.normal_stiffness
        )
        self.mass_unit = spec.density * self.length_unit**3
        self.force_unit = spec.normal_stiffness * self.length_unit**3
        self.stress_unit = spec.normal_stiffness * self.length_unit
        self.energy_unit = self.force_unit * self.length_unit
        self.ids = np.asarray(shape_ids, np.uint32)
        self.scales = np.asarray(scales, float)
        l = self.length_unit
        self.body = np.array(
            [
                [
                    s,
                    spec.density
                    * library.shapes[int(i)].volume
                    * s**3
                    / self.mass_unit,
                    library.shapes[int(i)].radius * s / l,
                    (6 * library.shapes[int(i)].volume / np.pi) ** (1 / 3) * s / l,
                ]
                for i, s in zip(self.ids, self.scales)
            ],
            np.float32,
        )
        inertia = np.array(
            [
                np.linalg.inv(
                    library.shapes[int(i)].inertia
                    * spec.density
                    * s**5
                    / (self.mass_unit * l * l)
                )
                for i, s in zip(self.ids, self.scales)
            ]
        )
        self.skin = float(spec.skin * 2 * self.body[:, 2].max())
        self.cutoff = float(2 * self.body[:, 2].max() + self.skin)
        box = np.asarray(cell, float) / l
        if spec.periodic and self.cutoff >= 0.5 * box.min():
            raise ValueError("Cell too small for unique periodic images")
        self.bins = np.maximum(1, np.minimum(64, np.floor(box / self.cutoff))).astype(
            np.uint32
        )
        self.bin_count = int(self.bins.prod())
        self.pair_capacity = max(1, self.n * spec.pairs_per_particle)
        self.contact_capacity = max(16, self.n * spec.contacts_per_particle)
        self.hash_size = 1 << math.ceil(math.log2(max(32, 2 * self.contact_capacity)))
        self.bin_capacity = spec.bin_capacity
        self._preflight_dt()
        estimated = (
            244 * self.contact_capacity
            + 8 * self.hash_size
            + 64 * self.pair_capacity
            + 8 * self.bin_count * self.bin_capacity
            + 1152 * self.n
        )
        estimated += 160 * self.wall_count * self.n + 256 * self.wall_count
        if spec.contact_geometry == "mesh":
            estimated += sum(
                224 * len(s.faces) + 224 + 8 * len(s.samples) for s in library.shapes
            )
        estimated += sum(s.sdf.nbytes + 32 * len(s.samples) for s in library.shapes)
        if estimated > spec.working_memory_gib * 1024**3:
            raise MemoryError(
                f"Estimated working set {estimated / 1024**3:.2f} GiB exceeds budget"
            )
        self.estimated_bytes = estimated
        self.buffers = {}
        self._static_geometry()
        self.add("shapeId", self.ids)
        self.add("body", self.body)
        self.add("invI", f4(inertia.reshape(-1, 3)))
        normalized_position = np.asarray(positions, dtype=float) / l
        self.add("position", f4(normalized_position))
        self.add("velocity", np.zeros((self.n, 4), np.float32))
        self.add("rotation", np.tile(np.array([0, 0, 0, 1], np.float32), (self.n, 1)))
        for name in (
            "momentum",
            "force",
            "torque",
            "referencePosition",
            "particleDiagnostic",
        ):
            self.add(name, np.zeros((self.n, 4), np.float32))
        self.add("particleStress", np.zeros((3 * self.n, 4), np.float32))
        self.add("diagnosticRows", np.zeros((8 * self.n, 4), np.float32))
        self.add("accumulated", np.zeros((self.n, 4), np.float32))
        for name in DAMPING_STATE:
            self.add(name, np.zeros((self.n, 4), np.float32))
        for name in RESIDUAL_STATE:
            residual = np.zeros((2 if name == "cellLog" else self.n, 4), np.float32)
            if name == "positionRemainder":
                residual[:, :3] = normalized_position - normalized_position.astype(
                    np.float32
                ).astype(float)
            self.add(name, residual)
        self.add("referencePositionRemainder", np.zeros((self.n, 4), np.float32))
        self.allocate("fabricRows", 2 * self.pair_capacity, 16)
        self.add(
            "cell",
            np.array([list(box) + [0], [0, 0, 0, 0], list(box) + [0]], np.float32),
        )
        self.initial_cell = np.asarray(cell, float)
        if spec.boundary_box is not None:
            self.initial_cell = box_extents(
                np.array([p["offset"] for p in spec.planes]), spec.boundary_box, cell
            )
        wall_plane = np.zeros((max(1, self.wall_count), 4), np.float64)
        wall_velocity = np.zeros_like(wall_plane)
        wall_material = np.zeros_like(wall_plane)
        for index, plane in enumerate(spec.planes):
            wall_plane[index] = [*plane["normal"], plane["offset"] / l]
            wall_velocity[index, :3] = np.array(plane["velocity"]) * self.time_unit / l
            wall_material[index, :3] = [
                plane["normal_stiffness"] / spec.normal_stiffness,
                plane["tangential_stiffness"] / spec.normal_stiffness,
                plane["friction"],
            ]
            wall_material[index, 3] = plane.get("normal_viscosity", 0) / (
                spec.normal_stiffness * self.time_unit
            )
        rounded_plane = wall_plane.astype(np.float32)
        residual_plane = (wall_plane - rounded_plane).astype(np.float32)
        if (
            not np.isfinite(rounded_plane).all()
            or not np.isfinite(residual_plane).all()
            or not np.isfinite(wall_velocity.astype(np.float32)).all()
            or not np.isfinite(wall_material.astype(np.float32)).all()
            or (
                self.wall_count
                and np.any(wall_material[: self.wall_count, :2].astype(np.float32) <= 0)
            )
        ):
            raise ValueError(
                "Plane data cannot be represented in normalized FP32 units"
            )
        cn = spec.normal_viscosity / (spec.normal_stiffness * self.time_unit)
        if not np.isfinite(np.float32(cn)) or (cn > 0 and np.float32(cn) <= 0):
            raise ValueError(
                "Normal viscosity cannot be represented in normalized FP32 units"
            )
        for index, plane in enumerate(spec.planes):
            if (
                plane.get("normal_viscosity", 0) > 0
                and np.float32(wall_material[index, 3]) <= 0
            ):
                raise ValueError(
                    "Plane viscosity cannot be represented in normalized FP32 units"
                )
        self.add("wallPlane", rounded_plane)
        self.add("wallRemainder", residual_plane)
        self.add("wallVelocity", wall_velocity.astype(np.float32))
        self.add("wallMaterial", wall_material.astype(np.float32))
        self.add("wallBoxIds", np.array(spec.boundary_box or [0] * 6, np.uint32))
        self.add(
            "wallAccumulated",
            np.zeros((max(1, self.wall_count * self.n), 4), np.float32),
        )
        self.add(
            "wallRows", np.zeros((max(1, 3 * self.wall_count * self.n), 4), np.float32)
        )
        self.add(
            "referenceCell",
            np.array([list(box) + [0], [0, 0, 0, self.cutoff]], np.float32),
        )
        ctrl = np.zeros(16, np.uint32)
        ctrl[4] = 1
        self.add("control", ctrl)
        for name, count in (
            ("binCounts", self.bin_count),
            ("binParticles", self.bin_count * self.bin_capacity),
            ("pairCounts", self.n),
            ("pairOffsets", self.n),
            ("contactCounts", self.pair_capacity),
            ("contactOffsets", self.pair_capacity),
            ("heads", self.n),
            ("next", 2 * self.contact_capacity),
        ):
            self.allocate(name, count)
        self.allocate("pairs", self.pair_capacity, 8)
        self.history_buffers = []
        for side in (0, 1):
            self.history_buffers.append(
                {
                    "meta": self.m.buffer(count=self.contact_capacity, stride=16),
                    "slip": self.m.buffer(count=self.contact_capacity, stride=16),
                    "normalArea": self.m.buffer(count=self.contact_capacity, stride=16),
                    "hash": self.m.buffer(
                        np.full(self.hash_size, 0xFFFFFFFF, np.uint32)
                    ),
                }
            )
        self.allocate("pointDepth", self.contact_capacity, 16)
        self.allocate("contactForce", self.contact_capacity, 16)
        self.allocate("normalLossPower", self.contact_capacity)
        self.scan_pairs = Scan(self.m, self.n)
        self.scan_contacts = Scan(self.m, self.pair_capacity)
        self.diag_buffers = []
        size = self.n
        while size > 1:
            size = (size + 255) // 256
            self.diag_buffers.append((size, self.m.buffer(count=size * 8, stride=16)))
        self.fabric_buffers = []
        size = self.pair_capacity
        while size > 1:
            size = (size + 255) // 256
            self.fabric_buffers.append((size, self.m.buffer(count=size * 2, stride=16)))
        self.wall_diag_buffers = []
        size = self.n
        while self.wall_count and size > 1:
            size = (size + 255) // 256
            self.wall_diag_buffers.append(
                (size, self.m.buffer(count=3 * size * self.wall_count, stride=16))
            )
        names = (
            "position",
            "velocity",
            "rotation",
            "momentum",
            "cell",
            "force",
            "torque",
            "referencePosition",
            "referenceCell",
            "control",
            "pairs",
            "pairCounts",
            "pairOffsets",
            "particleStress",
            "particleDiagnostic",
            "accumulated",
            "referencePositionRemainder",
            *RESIDUAL_STATE,
            *DAMPING_STATE,
            *(WALL_STATE if self.wall_count else ()),
        )
        self.backups = {
            name: self.m.buffer(count=self.buffers[name].size // 4) for name in names
        }
        self.history_backup = {
            name: self.m.buffer(count=self.hash_size if name == "hash" else 4)
            for name in ("meta", "slip", "normalArea", "hash")
        }

    def _preflight_dt(self):
        limits = []
        for i, s in zip(self.ids, self.scales):
            sh = self.library.shapes[int(i)]
            mass = self.spec.density * sh.volume * s**3
            stiffness = (
                (
                    self.spec.normal_stiffness
                    + self.spec.tangential_stiffness
                    + sum(
                        p["normal_stiffness"] + p["tangential_stiffness"]
                        for p in self.spec.planes
                    )
                )
                * sh.areas.sum()
                * s
                * s
            )
            rotation = np.linalg.eigvalsh(
                sh.inertia * self.spec.density * s**5
            ).min() / (stiffness * (sh.radius * s) ** 2)
            elastic_limit = math.sqrt(min(mass / stiffness, rotation))
            viscosity = (
                (
                    self.spec.normal_viscosity
                    + sum(p.get("normal_viscosity", 0) for p in self.spec.planes)
                )
                * sh.areas.sum()
                * s**2
            )
            viscous_limit = math.inf
            if viscosity > 0:
                viscous_limit = min(
                    mass / viscosity,
                    np.linalg.eigvalsh(sh.inertia * self.spec.density * s**5).min()
                    / (viscosity * (sh.radius * s) ** 2),
                )
            limits.append(0.1 * min(elastic_limit, viscous_limit))
        if self.spec.timestep > min(limits):
            raise ValueError(
                f"Timestep exceeds conservative estimate {min(limits):.6g} s"
            )

    def add(self, name, array):
        self.buffers[name] = self.m.buffer(array)

    def allocate(self, name, count, stride=4):
        self.buffers[name] = self.m.buffer(count=count, stride=stride)

    def _static_geometry(self):
        if self.spec.contact_geometry == "mesh":
            from .mesh import pack_meshes

            mesh_data = pack_meshes(self.library, self.length_unit)
        else:
            mesh_data = {
                name: np.zeros(
                    (1, 4),
                    np.uint32
                    if name
                    in {
                        "meshRoots",
                        "meshLinks",
                        "meshHullInfo",
                        "meshClusterInfo",
                        "meshClusterRanges",
                        "meshSampleIndices",
                    }
                    else np.float32,
                )
                for name in (
                    "meshHullInfo",
                    "meshPlanes",
                    "meshRoots",
                    "meshLinks",
                    "meshMin",
                    "meshMax",
                    "meshTriangles",
                    "meshNormals",
                    "meshClusterInfo",
                    "meshClusterRanges",
                    "meshClusterMin",
                    "meshClusterMax",
                    "meshSampleIndices",
                )
            }
        for name, data in mesh_data.items():
            self.add(name, data)
        grids = []
        info = []
        origins = []
        fields = []
        samples = []
        go = so = 0
        l = self.length_unit
        for s in self.library.shapes:
            grids.append([go, *s.sdf.shape])
            info.append([so, len(s.samples), 0, 0])
            origins.append([*(s.origin / l), s.spacing / l])
            fields.append((s.sdf / l).ravel())
            samples.append(
                np.column_stack([s.samples / l, s.areas / l**2]).astype(np.float32)
            )
            go += s.sdf.size
            so += len(s.samples)
        for name, array in [
            ("grids", np.asarray(grids, np.uint32)),
            ("sampleInfo", np.asarray(info, np.uint32)),
            ("gridOrigin", np.asarray(origins, np.float32)),
            ("distance", np.concatenate(fields).astype(np.float32)),
            ("sample", np.concatenate(samples)),
        ]:
            self.add(name, array)

    def bindings(self, dt=None, drag=0):
        new = self.history_buffers[self.active]
        old = self.history_buffers[1 - self.active]
        c = {
            "N": self.n,
            "shapes": len(self.library.shapes),
            "pairCapacity": self.pair_capacity,
            "contactCapacity": self.contact_capacity,
            "binCapacity": self.bin_capacity,
            "binCount": self.bin_count,
            "bins": self.bins.tolist(),
            "hashSize": self.hash_size,
            "dt": (self.spec.timestep if dt is None else dt) / self.time_unit,
            "kn": 1.0,
            "kt": self.spec.tangential_stiffness / self.spec.normal_stiffness,
            "mu": self.spec.friction,
            "normalViscosity": self.spec.normal_viscosity
            / (self.spec.normal_stiffness * self.time_unit),
            "skin": self.skin,
            "drag": drag * self.time_unit,
            "maxOverlap": self.spec.max_overlap_fraction,
            "historyStep": int(self.history),
            "meshGeometry": int(self.spec.contact_geometry == "mesh"),
            "periodic": int(self.spec.periodic),
            "wallCount": self.wall_count,
            "wallBoxEnabled": int(self.spec.boundary_box is not None),
        }
        return dict(
            self.buffers,
            c=c,
            **new,
            oldMeta=old["meta"],
            oldSlip=old["slip"],
            oldNormal=old["normalArea"],
            oldHash=old["hash"],
        )

    def dispatch(self, name, n, e, b):
        self.m.dispatch(name, n, e, r=b)

    def _encode_contacts(self, e, b):
        if self.spec.contact_geometry == "mesh":
            self.dispatch("resetMeshContacts", self.pair_capacity, e, b)
            self.dispatch("collectMeshContacts", 64 * self.pair_capacity, e, b)
        else:
            self.dispatch("countContacts", 64 * self.pair_capacity, e, b)
            self.scan_contacts.encode(
                e, self.buffers["contactCounts"], self.buffers["contactOffsets"]
            )
            self.dispatch("contactTotal", 1, e, b)
            self.dispatch("emitContacts", 64 * self.pair_capacity, e, b)
        if self.wall_count:
            self.dispatch("collectWallContacts", 64 * self.n * self.wall_count, e, b)

    def forces(self, e, b, *, capture_kinematics=True):
        if self.viscous and capture_kinematics:
            self.dispatch("captureContactKinematics", self.n, e, b)
        self.dispatch("clearBins", self.bin_count, e, b)
        self.dispatch("fillBins", self.n, e, b)
        self.dispatch("countPairs", self.n, e, b)
        self.scan_pairs.encode(
            e, self.buffers["pairCounts"], self.buffers["pairOffsets"]
        )
        self.dispatch("emitPairs", self.n, e, b)
        self._encode_contacts(e, b)
        self.dispatch("clearContactStorage", max(self.n, self.hash_size), e, b)
        self.dispatch("resolveContacts", self.contact_capacity, e, b)
        self.dispatch("buildHistoryHash", self.contact_capacity, e, b)
        self.dispatch("gather", self.n, e, b)
        if self.wall_count:
            self.dispatch("gatherWallRows", self.n * self.wall_count, e, b)

    def initialize(self):
        e = self.m.device.create_command_encoder()
        self.forces(e, self.bindings(dt=0))
        self.m.submit(e)
        self.m.device.wait()
        self.check_errors()
        self.history = True
        self.active = 1 - self.active

    def check_errors(self):
        ctrl = self.m.read(self.buffers["control"], np.uint32)
        if ctrl[15] != 0:
            raise RuntimeError("Arithmetic materialization mask must remain zero")
        if ctrl[0]:
            raise RejectedStep(ctrl[0])
        return ctrl

    def advance(self, steps, *, rate=(0, 0, 0), drag=0, wall_velocities=None):
        from dataclasses import replace

        if type(steps) is not int or steps < 0:
            raise ValueError("Nonnegative integer step count required")
        rate = finite_array(rate, (3,))
        if not self.spec.periodic and np.any(rate != 0):
            raise ValueError("Affine cell rates require a periodic domain")
        if not np.isfinite(drag) or drag < 0:
            raise ValueError("Finite nonnegative drag required")
        if wall_velocities is not None:
            if not self.wall_count:
                raise ValueError("Wall velocities require planes")
            wall_velocities = finite_array(wall_velocities, (self.wall_count, 3))
        for attempt in range(5):
            try:
                return self._advance_once(
                    steps, rate=rate, drag=drag, wall_velocities=wall_velocities
                )
            except RejectedStep as exc:
                capacity = exc.flags & (8 | 16 | 32 | 64)
                if not capacity or exc.flags & ~(8 | 16 | 32 | 64) or attempt == 4:
                    raise
                state = self.state()
                old = self.spec
                settings = {}
                if capacity & 8:
                    settings["bin_capacity"] = old.bin_capacity * 2
                if capacity & 16:
                    settings["pairs_per_particle"] = old.pairs_per_particle * 2
                if capacity & (32 | 64):
                    settings["contacts_per_particle"] = max(
                        old.contacts_per_particle * 2,
                        math.ceil(2 * self.contact_capacity / self.n),
                    )
                log = self.capacity_retries + [
                    {"step": self.step, "flags": exc.flags, "new_capacities": settings}
                ]
                replacement = Engine(
                    self.library,
                    replace(old, **settings),
                    state["position"][:, :3] * self.length_unit,
                    state["cell"][0, :3] * self.length_unit,
                    self.ids,
                    self.scales,
                    metal=self.m,
                )
                replacement.initial_cell = self.initial_cell.copy()
                replacement.restore(state, self.step)
                self.__dict__.update(replacement.__dict__)
                self.capacity_retries = log

    def _reserve_history_backup(self):
        """Reserve only the committed prefix; unused history rows are unreachable."""
        count = int(self.m.read(self.buffers["control"], np.uint32)[2])
        if count < 0 or count > self.contact_capacity:
            raise ValueError("Committed contact count is outside capacity")
        size = max(16, count * 16)
        replacements = {
            name: self.m.buffer(count=size // 4)
            for name in ("meta", "slip", "normalArea")
            if self.history_backup[name].size < size
        }
        self.history_backup.update(replacements)
        return count

    def _copy_history_backup(self, encoder, committed, count, *, restore=False):
        for name, backup in self.history_backup.items():
            size = committed[name].size if name == "hash" else count * 16
            if size:
                source, destination = (
                    (backup, committed[name]) if restore else (committed[name], backup)
                )
                encoder.copy_buffer(destination, 0, source, 0, size)

    def _advance_once(self, steps, *, rate=(0, 0, 0), drag=0, wall_velocities=None):
        if getattr(self, "invalid", False):
            raise RuntimeError("Engine is invalid; restore a checkpoint")
        if steps < 1:
            return
        # Reserve before mutating cell/wall state so allocation failure is atomic.
        history_count = self._reserve_history_backup()
        cell = self.m.read(self.buffers["cell"], shape=(3, 4))
        original_cell = cell.copy()
        original_wall_velocity = None
        if wall_velocities is not None:
            original_wall_velocity = self.m.read(
                self.buffers["wallVelocity"], shape=(self.wall_count, 4)
            )
            normalized_velocity = f4(
                wall_velocities * self.time_unit / self.length_unit
            )
            if not np.isfinite(normalized_velocity).all():
                raise ValueError(
                    "Wall velocities cannot be represented in normalized FP32 units"
                )
            self.buffers["wallVelocity"].copy_from_numpy(normalized_velocity)
        cell[1, :3] = np.asarray(rate) * self.time_unit
        self.buffers["cell"].copy_from_numpy(cell)
        e = self.m.device.create_command_encoder()
        for name, backup in self.backups.items():
            e.copy_buffer(backup, 0, self.buffers[name], 0, self.buffers[name].size)
        original_active = self.active
        original_history = self.history
        committed = self.history_buffers[1 - self.active]
        self._copy_history_backup(e, committed, history_count)
        if original_wall_velocity is not None:
            if any(
                p.get("normal_viscosity", 0) > 0 for p in self.spec.planes
            ) and not np.array_equal(original_wall_velocity, normalized_velocity):
                self.forces(e, self.bindings(dt=0))
                self.active = 1 - self.active
                self.history = True
            else:
                self.dispatch(
                    "refreshWallPower", self.n * self.wall_count, e, self.bindings()
                )
        for _ in range(steps):
            b = self.bindings(drag=drag)
            self.dispatch("beginStep", self.n, e, b)
            self.dispatch("advanceCell", 1, e, b)
            if self.wall_count:
                self.dispatch("advanceWalls", 1, e, b)
            self.dispatch("checkRebuild", self.n, e, b)
            self.forces(e, b)
            self.dispatch("finishStep", self.n, e, b)
            self.active = 1 - self.active
            self.history = True
        self.m.submit(e)
        self.m.device.wait()
        try:
            self.check_errors()
        except RuntimeError:
            e = self.m.device.create_command_encoder()
            for name, backup in self.backups.items():
                e.copy_buffer(self.buffers[name], 0, backup, 0, backup.size)
            self._copy_history_backup(e, committed, history_count, restore=True)
            self.m.submit(e)
            self.m.device.wait()
            self.buffers["cell"].copy_from_numpy(original_cell)
            if original_wall_velocity is not None:
                self.buffers["wallVelocity"].copy_from_numpy(original_wall_velocity)
            self.active = original_active
            self.history = original_history
            raise
        self.step += steps

    def diagnostics(self):
        e = self.m.device.create_command_encoder()
        self.dispatch("diagnosticRows", self.n, e, self.bindings())
        source = self.buffers["diagnosticRows"]
        n = self.n
        for size, dest in self.diag_buffers:
            self.m.kernel("reduceRows", "diagnostic.slang").dispatch(
                thread_count=[size * 256, 1, 1],
                command_encoder=e,
                N=n,
                input=source,
                output=dest,
            )
            source = dest
            n = size
        self.dispatch("fabricRows", self.pair_capacity, e, self.bindings())
        fabric_source = self.buffers["fabricRows"]
        n = self.pair_capacity
        for size, dest in self.fabric_buffers:
            self.m.kernel("reduceFabric", "diagnostic.slang").dispatch(
                thread_count=[size * 256, 1, 1],
                command_encoder=e,
                N=n,
                input=fabric_source,
                output=dest,
            )
            fabric_source = dest
            n = size
        wall_source = self.buffers["wallRows"]
        n = self.n
        for size, dest in self.wall_diag_buffers:
            self.m.kernel("reduceWallRows", "diagnostic.slang").dispatch(
                thread_count=[size * 256, self.wall_count, 1],
                command_encoder=e,
                N=n,
                outputCount=size,
                input=wall_source,
                output=dest,
            )
            wall_source = dest
            n = size
        self.m.submit(e)
        self.m.device.wait()
        data = self.m.read(source, shape=(8, 4))
        fabric_data = self.m.read(fabric_source, shape=(2, 4))
        active_pairs = round(float(fabric_data[1, 2]))
        fabric = np.array(
            [
                [fabric_data[0, 0], fabric_data[0, 3], fabric_data[1, 0]],
                [fabric_data[0, 3], fabric_data[0, 1], fabric_data[1, 1]],
                [fabric_data[1, 0], fabric_data[1, 1], fabric_data[0, 2]],
            ]
        ) / max(active_pairs, 1)
        cell = self.m.read(self.buffers["cell"], shape=(3, 4))
        ctrl = self.check_errors()
        box = cell[0, :3].astype(float) * self.length_unit
        domain = box.copy()
        wall_results = []
        if self.wall_count:
            wall_data = self.m.read(wall_source, shape=(self.wall_count, 3, 4))
            plane = self.m.read(
                self.buffers["wallPlane"], shape=(self.wall_count, 4)
            ).astype(float)
            remainder = self.m.read(
                self.buffers["wallRemainder"], shape=(self.wall_count, 4)
            ).astype(float)
            wall_velocity = self.m.read(
                self.buffers["wallVelocity"], shape=(self.wall_count, 4)
            )
            offsets = (plane[:, 3] + remainder[:, 3]) * self.length_unit
            for w in range(self.wall_count):
                wall_results.append(
                    dict(
                        index=w,
                        normal=plane[w, :3].tolist(),
                        offset=float(offsets[w]),
                        velocity=(
                            wall_velocity[w, :3] * self.length_unit / self.time_unit
                        ).tolist(),
                        reaction=(wall_data[w, 0, :3] * self.force_unit).tolist(),
                        reaction_moment_origin=(
                            wall_data[w, 1, :3] * self.energy_unit
                        ).tolist(),
                        active_surface_contacts=int(wall_data[w, 0, 3].view(np.uint32)),
                        power=float(
                            wall_data[w, 1, 3] * self.energy_unit / self.time_unit
                        ),
                        work=float(wall_data[w, 2, 0] * self.energy_unit),
                    )
                )
            if self.spec.boundary_box is not None:
                box = box_extents(offsets, self.spec.boundary_box, domain)
        volume = float(box.prod())
        contact = data[:3, :3].astype(float) * self.energy_unit / volume
        kinetic = data[3:6, :3].astype(float) * self.energy_unit / volume
        total = contact + kinetic
        symmetric = 0.5 * (total + total.T)
        radial = 0.5 * (symmetric[0, 0] + symmetric[1, 1])
        solid = sum(
            self.library.shapes[int(i)].volume * s**3
            for i, s in zip(self.ids, self.scales)
        )
        return {
            "step": self.step,
            "elapsed_time": self.step * self.spec.timestep,
            "cell": box.tolist(),
            "domain_cell": domain.tolist(),
            "walls": wall_results,
            "stress": total.tolist(),
            "contact_stress": contact.tolist(),
            "kinetic_stress": kinetic.tolist(),
            "stress_antisymmetry": float(
                np.linalg.norm(total - total.T) / max(np.linalg.norm(total), 1e-30)
            ),
            "p": float(np.trace(symmetric) / 3),
            "q": float(symmetric[2, 2] - radial),
            "axial_strain": float(1 - box[2] / self.initial_cell[2]),
            "volumetric_strain": float(np.sum(1 - box / self.initial_cell)),
            "volume_ratio": float(volume / np.prod(self.initial_cell)),
            "log_strain": (-np.log(box / self.initial_cell)).tolist(),
            "void_ratio": float(volume / solid - 1),
            "kinetic_energy": float(data[0, 3] * self.energy_unit),
            "elastic_energy_proxy": float(data[1, 3] * self.energy_unit),
            "unbalanced_force": float(data[2, 3] / max(data[3, 3], 1e-30)),
            "max_overlap_fraction": float(data[4, 3]),
            "friction_dissipation_last_step": float(data[5, 3] * self.energy_unit),
            "candidate_pairs": int(ctrl[1]),
            "active_surface_contacts": int(ctrl[2]),
            "active_grain_pairs": active_pairs,
            "coordination": 2 * active_pairs / self.n,
            "fabric": fabric.tolist(),
            "sliding_fraction": float(ctrl[6] / max(ctrl[2], 1)),
            "cell_log_rate": (cell[1, :3] / self.time_unit).tolist(),
            "friction_dissipation": float(data[6, 0] * self.energy_unit),
            "drag_dissipation": float(data[6, 1] * self.energy_unit),
            "cell_work": float(data[6, 2] * self.energy_unit),
            "wall_work": float(data[6, 3] * self.energy_unit),
            "normal_contact_dissipation": float(data[7, 0] * self.energy_unit),
            "normal_contact_power": float(
                data[7, 1] * self.energy_unit / self.time_unit
            ),
        }

    def state(self):
        out = {
            name: self.m.read(
                self.buffers[name],
                shape=(
                    (3 if name == "cell" else 2 if name == "cellLog" else self.n),
                    4,
                ),
            )
            for name in (
                "position",
                "velocity",
                "rotation",
                "momentum",
                "cell",
                "force",
                "torque",
                "accumulated",
                *RESIDUAL_STATE,
            )
        }
        if self.viscous:
            for name in DAMPING_STATE:
                out[name] = self.m.read(self.buffers[name], shape=(self.n, 4))
        if self.wall_count:
            for name in WALL_STATE:
                size = (
                    self.wall_count * self.n * (3 if name == "wallRows" else 1)
                    if name in {"wallRows", "wallAccumulated"}
                    else self.wall_count
                )
                out[name] = self.m.read(self.buffers[name], shape=(size, 4))
        ctrl = self.check_errors()
        h = self.history_buffers[1 - self.active]
        count = int(ctrl[2])
        out["contacts"] = count
        for name in ("meta", "slip", "normalArea"):
            out[name] = self.m.read_prefix(
                h[name],
                4 * count,
                np.uint32 if name == "meta" else np.float32,
                shape=(count, 4),
            )
        return out

    def profile_step(self):
        """One real step, timestamped by subsystem; benchmark-only synchronization."""
        pool = self.m.device.create_query_pool(self.m.spy.QueryType.timestamp, 12)
        e = self.m.device.create_command_encoder()
        b = self.bindings()
        e.write_timestamp(pool, 0)
        self.dispatch("beginStep", self.n, e, b)
        self.dispatch("advanceCell", 1, e, b)
        if self.wall_count:
            self.dispatch("advanceWalls", 1, e, b)
        self.dispatch("checkRebuild", self.n, e, b)
        e.write_timestamp(pool, 1)
        self.dispatch("clearBins", self.bin_count, e, b)
        self.dispatch("fillBins", self.n, e, b)
        self.dispatch("countPairs", self.n, e, b)
        self.scan_pairs.encode(
            e, self.buffers["pairCounts"], self.buffers["pairOffsets"]
        )
        self.dispatch("emitPairs", self.n, e, b)
        e.write_timestamp(pool, 2)
        if self.viscous:
            self.dispatch("captureContactKinematics", self.n, e, b)
        self._encode_contacts(e, b)
        e.write_timestamp(pool, 3)
        self.dispatch("clearContactStorage", max(self.n, self.hash_size), e, b)
        self.dispatch("resolveContacts", self.contact_capacity, e, b)
        self.dispatch("buildHistoryHash", self.contact_capacity, e, b)
        e.write_timestamp(pool, 4)
        self.dispatch("gather", self.n, e, b)
        if self.wall_count:
            self.dispatch("gatherWallRows", self.n * self.wall_count, e, b)
        e.write_timestamp(pool, 5)
        self.dispatch("finishStep", self.n, e, b)
        e.write_timestamp(pool, 6)
        self.m.submit(e)
        self.m.device.wait()
        self.check_errors()
        self.active = 1 - self.active
        self.step += 1
        try:
            ticks = pool.get_results(0, 7)
        except RuntimeError as exc:
            return {
                "available": False,
                "reason": str(exc),
                "timing_source": "GPU timestamps unavailable; use separately measured wall time",
            }
        freq = self.m.device.info.timestamp_frequency
        names = (
            "drift",
            "neighbours",
            "contact_geometry",
            "friction_and_history",
            "force_reduction",
            "final_kick",
        )
        return {
            "timestamp_frequency_hz": freq,
            "seconds": {
                name: (ticks[i + 1] - ticks[i]) / freq for i, name in enumerate(names)
            },
            "total_gpu_seconds": (ticks[-1] - ticks[0]) / freq,
        }

    def restore(self, state, step):
        if self.viscous:
            for name in DAMPING_STATE:
                finite_array(state[name], (self.n, 4))
        if self.wall_count:
            for name in WALL_STATE:
                value = finite_array(state[name])
                if value.shape != (self.buffers[name].size // 16, 4):
                    raise ValueError("Checkpoint wall state shape mismatch")
            normals = np.array(
                [plane["normal"] for plane in self.spec.planes], np.float32
            )
            if not np.array_equal(state["wallPlane"][:, :3], normals):
                raise ValueError("Checkpoint plane normal identity mismatch")
            offsets = (
                np.asarray(state["wallPlane"][:, 3], float)
                + state["wallRemainder"][:, 3]
            ) * self.length_unit
            if self.spec.boundary_box is not None:
                box_extents(
                    offsets,
                    self.spec.boundary_box,
                    np.asarray(state["cell"][0, :3], float) * self.length_unit,
                )
            for name in WALL_STATE:
                self.buffers[name].copy_from_numpy(state[name])
        for name in DAMPING_STATE:
            self.buffers[name].copy_from_numpy(
                state[name] if self.viscous else np.zeros((self.n, 4), np.float32)
            )
        for name in (
            "position",
            "velocity",
            "rotation",
            "momentum",
            "cell",
            "force",
            "torque",
            "accumulated",
            *RESIDUAL_STATE,
        ):
            self.buffers[name].copy_from_numpy(state[name])
        count = int(state["contacts"])
        if count < 0 or count > self.contact_capacity:
            raise ValueError("Checkpoint contact count is outside capacity")
        self.active = 0
        h = self.history_buffers[1]
        for name in ("meta", "slip", "normalArea"):
            values = np.asarray(
                state[name], dtype=np.uint32 if name == "meta" else np.float32
            )
            if values.shape != (count, 4):
                raise ValueError("Checkpoint contact history shape mismatch")
            self.m.write_prefix(h[name], values)
        ctrl = np.zeros(16, np.uint32)
        ctrl[2] = count
        ctrl[4] = 1
        self.buffers["control"].copy_from_numpy(ctrl)
        self.active = 1
        self.history = True
        e = self.m.device.create_command_encoder()
        b = self.bindings(dt=0)
        self.dispatch("clearContactStorage", max(self.n, self.hash_size), e, b)
        self.dispatch("buildHistoryHash", self.contact_capacity, e, b)
        self.m.submit(e)
        self.m.device.wait()
        self.check_errors()
        self.active = 0
        ctrl[4] = 1
        self.buffers["control"].copy_from_numpy(ctrl)
        e = self.m.device.create_command_encoder()
        self.forces(e, self.bindings(dt=0), capture_kinematics=False)
        self.m.submit(e)
        self.m.device.wait()
        self.check_errors()
        self.active = 1
        self.step = step
