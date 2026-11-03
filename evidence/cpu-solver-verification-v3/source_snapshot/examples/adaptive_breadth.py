"""Bounded adaptive triangle octrees; experimental exact-feature acceleration."""

import hashlib
import time
from collections import deque

import numpy as np
from brick_mesh import brick_shader as uniform_shader
from scipy.spatial import cKDTree


def build_bricks(roots, links, triangles, *, resolution, capacity):
    # Resolution encodes the maximum dyadic subdivision, not a dense directory.
    max_depth = int(np.log2(resolution))
    assert 2**max_depth == resolution and max_depth <= 8
    budget = 100000
    started = time.perf_counter()
    info, centers, halves, ranges, indices, reports = [], [], [], [], [], []
    for sid, root in enumerate(roots):
        leaves = links[int(root[0]) : int(root[1])]
        leaves = leaves[leaves[:, 2] > 0]
        first = int(leaves[:, 1].min())
        end = int((leaves[:, 1] + leaves[:, 2]).max())
        tri = triangles[first:end, :, :3].astype(float)
        tlo, thi = tri.min(axis=1), tri.max(axis=1)
        vertex_tree = cKDTree(tri.reshape(-1, 3))
        margin = 128 * np.finfo(np.float32).eps * (1 + abs(tri).max())
        low, high = tlo.min(axis=0), thi.max(axis=0)
        root_center = ((low + high) * 0.5).astype(np.float32)
        root_half = ((high - low) * 0.5).astype(np.float32)
        base = len(ranges)
        info.append([base, max_depth, capacity, 0])
        stats = {
            "triangles": end - first,
            "nodes": 0,
            "list_leaves": 0,
            "fallback_leaves": 0,
            "max_depth_fallbacks": 0,
            "budget_fallbacks": 0,
            "interior_fallbacks": 0,
            "deepest_level": 0,
        }

        def reserve(center, half, margin=margin):
            node = len(ranges)
            radius = np.nextafter(
                np.float32(np.linalg.norm(half.astype(float)) + margin),
                np.float32(np.inf),
            )
            centers.append([*center, radius])
            halves.append([*half, 0])
            ranges.append([0, 0, 0, 0])
            return node

        reserve(root_center, root_half)
        pending = deque([(base, np.arange(end - first), 0)])

        def populate(
            node,
            ids,
            depth,
            vertex_tree=vertex_tree,
            tlo=tlo,
            thi=thi,
            margin=margin,
            stats=stats,
            first=first,
            base=base,
            reserve=reserve,
            pending=pending,
        ):
            c = np.asarray(centers[node][:3], float)
            half = np.asarray(halves[node][:3], np.float32)
            radius = float(centers[node][3])
            upper = float(vertex_tree.query(c)[0])
            d = np.maximum(np.maximum(tlo[ids] - c, c - thi[ids]), 0)
            d2 = np.einsum("ij,ij->i", d, d)
            ids = ids[d2 <= (upper + 2 * radius + margin) ** 2]
            stats["deepest_level"] = max(stats["deepest_level"], depth)
            if 0 < len(ids) <= capacity:
                ranges[node] = [len(indices), len(ids), 0, 2]
                indices.extend((first + ids).tolist())
                stats["list_leaves"] += 1
                return
            reason = None
            if depth >= max_depth:
                reason = "max_depth_fallbacks"
            elif len(ranges) - base + 8 > budget:
                reason = "budget_fallbacks"
            elif len(d2) == 0 or float(d2.min()) > (1.5 * radius + margin) ** 2:
                reason = "interior_fallbacks"
            if reason is not None:
                ranges[node][3] = {
                    "interior_fallbacks": 3,
                    "budget_fallbacks": 4,
                    "max_depth_fallbacks": 5,
                }[reason]
                stats["fallback_leaves"] += 1
                stats[reason] += 1
                return
            child_base = len(ranges)
            ranges[node] = [0, 0, child_base, 1]
            child_half = (half * 0.5).astype(np.float32)
            for child in range(8):
                sign = np.array([1 if child & (1 << a) else -1 for a in range(3)])
                child_center = (c + sign * child_half).astype(np.float32)
                reserve(child_center, child_half)
            pending.extend((child_base + child, ids, depth + 1) for child in range(8))

        while pending:
            populate(*pending.popleft())
        stats["nodes"] = len(ranges) - base
        reports.append(stats)
        print(
            f"adaptive shape {sid + 1}/{len(roots)} nodes={stats['nodes']} lists={stats['list_leaves']} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    arrays = {
        name: np.asarray(values, dtype)
        for name, values, dtype in [
            ("brickInfo", info, np.uint32),
            ("brickCenters", centers, np.float32),
            ("brickHalf", halves, np.float32),
            ("brickRanges", ranges, np.uint32),
            ("brickTriangles", indices or [0], np.uint32),
        ]
    }
    return arrays, {
        "build_seconds": time.perf_counter() - started,
        "resolution": resolution,
        "capacity": capacity,
        "max_depth": max_depth,
        "node_budget_per_shape": budget,
        "per_shape": reports,
        "bytes": sum(a.nbytes for a in arrays.values()),
        "array_sha256": {
            n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in arrays.items()
        },
        "margin_eps_multiplier": 128,
    }


SELECTION = """float4 meshDistance(uint shape,float3 p){
    uint node=r.brickInfo[shape].x;
    uint4 list=0;
    for(uint level=0;level<9;level++){
        float4 center=r.brickCenters[node];
        if(length(p-center.xyz)>center.w)return meshDistanceBVH(shape,p);
        list=r.brickRanges[node];
        if(list.w==2)break;
        if(list.w!=1)return meshDistanceBVH(shape,p);
        uint child=(p.x>=center.x?1u:0u)|(p.y>=center.y?2u:0u)|(p.z>=center.z?4u:0u);
        node=list.z+child;
    }
    if(list.w!=2||list.y==0)return meshDistanceBVH(shape,p);
    """


def brick_shader(source, metrics=False):
    result = uniform_shader(source, metrics=metrics)
    result = result.replace(
        "    StructuredBuffer<float4> brickOrigin;\n    StructuredBuffer<float4> brickScale;",
        "    StructuredBuffer<float4> brickHalf;",
    )
    start = result.index("float4 meshDistance(uint shape,float3 p){")
    end = (
        result.index("    // BRICK_METRIC_LIST", start)
        if not metrics
        else result.index("    uint unused;InterlockedAdd(r.meshQueryMetrics[4]", start)
    )
    selection = SELECTION
    if metrics:
        selection = selection.replace(
            "if(length(p-center.xyz)>center.w)return meshDistanceBVH(shape,p);",
            "if(length(p-center.xyz)>center.w){uint unused;InterlockedAdd(r.meshQueryMetrics[11],1,unused);return meshDistanceBVH(shape,p);}",
        )
        selection = selection.replace(
            "if(list.w!=1)return meshDistanceBVH(shape,p);",
            "if(list.w!=1){uint unused;uint slot=list.w==4?7u:(list.w==5?10u:9u);InterlockedAdd(r.meshQueryMetrics[slot],1,unused);return meshDistanceBVH(shape,p);}",
        )
    return result[:start] + selection + result[end:]
