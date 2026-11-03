"""Conservative body-frame triangle bricks; experimental, no source repair."""

import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree


def build_bricks(roots, links, triangles, *, resolution, capacity):
    """Use nearest vertex upper bound U and spherical brick radius R.

    For p in the brick, d(p, mesh) <= U+R. Any winning triangle therefore
    has distance from the brick center <= U+2R. Triangle AABB filtering is
    conservative; excessive lists are marked for BVH fallback, never truncated.
    """
    started = time.perf_counter()
    info, origins, scales, centers, ranges, indices, counts = [], [], [], [], [], [], []
    for root in roots:
        leaves = links[int(root[0]) : int(root[1])]
        leaves = leaves[leaves[:, 2] > 0]
        first = int(leaves[:, 1].min())
        end = int((leaves[:, 1] + leaves[:, 2]).max())
        tri = triangles[first:end, :, :3].astype(float)
        low, high = tri.min(axis=(0, 1)), tri.max(axis=(0, 1))
        step = ((high - low) / resolution).astype(np.float32)
        low = low.astype(np.float32)
        inv = (1 / step).astype(np.float32)
        xyz = np.indices((resolution,) * 3).reshape(3, -1).T
        cs = (low.astype(float) + (xyz + 0.5) * step.astype(float)).astype(np.float32)
        margin = 128 * np.finfo(np.float32).eps * (1 + abs(tri).max())
        radius = np.nextafter(
            np.float32(np.linalg.norm(step.astype(float) * 0.5) + margin),
            np.float32(np.inf),
        )
        upper, _ = cKDTree(tri.reshape(-1, 3)).query(cs.astype(float))
        threshold = upper + 2 * float(radius) + margin
        tc = tri.mean(axis=1)
        tr = np.linalg.norm(tri - tc[:, None, :], axis=2).max(axis=1)
        candidates = cKDTree(tc).query_ball_point(
            cs.astype(float), threshold + tr.max() + margin
        )
        tlo, thi = tri.min(axis=1), tri.max(axis=1)
        info.append([len(ranges), resolution, 0, 0])
        origins.append([*low, 0])
        scales.append([*inv, 0])
        accepted = 0
        for c, u, ids in zip(cs, threshold, candidates):
            ids = np.asarray(ids, np.int64)
            d = np.maximum(np.maximum(tlo[ids] - c, c - thi[ids]), 0)
            ids = np.sort(ids[np.einsum("ij,ij->i", d, d) <= u * u])
            centers.append([*c, radius])
            if 0 < len(ids) <= capacity:
                ranges.append([len(indices), len(ids), 0, 0])
                indices.extend((first + ids).tolist())
                accepted += 1
            else:
                ranges.append([0, 0, 0, 0])
        counts.append(
            {"triangles": end - first, "bricks": len(cs), "usable_bricks": accepted}
        )
    arrays = {
        name: np.asarray(values, dtype)
        for name, values, dtype in [
            ("brickInfo", info, np.uint32),
            ("brickOrigin", origins, np.float32),
            ("brickScale", scales, np.float32),
            ("brickCenters", centers, np.float32),
            ("brickRanges", ranges, np.uint32),
            ("brickTriangles", indices or [0], np.uint32),
        ]
    }
    receipt = {
        "build_seconds": time.perf_counter() - started,
        "resolution": resolution,
        "capacity": capacity,
        "per_shape": counts,
        "bytes": sum(a.nbytes for a in arrays.values()),
        "array_sha256": {
            name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in arrays.items()
        },
        "margin_eps_multiplier": 128,
    }
    return arrays, receipt


WRAPPER = """
float4 meshDistance(uint shape,float3 p){
    uint4 info=r.brickInfo[shape];
    float3 g=(p-r.brickOrigin[shape].xyz)*r.brickScale[shape].xyz;
    if(any(g<0)||any(g>=float(info.y)))return meshDistanceBVH(shape,p);
    uint3 cell=uint3(floor(g));
    uint brick=info.x+(cell.x*info.y+cell.y)*info.y+cell.z;
    float4 center=r.brickCenters[brick];
    uint4 list=r.brickRanges[brick];
    if(list.y==0||length(p-center.xyz)>center.w)return meshDistanceBVH(shape,p);
    // BRICK_METRIC_LIST
    float best=1e30,second=1e30;float3 nearest=0,pseudo=0;
    for(uint k=0;k<list.y;k++){
        uint triangle=r.brickTriangles[list.x+k],feature;
        float3 q=closestTriangle(p,r.meshTriangles[3*triangle].xyz,r.meshTriangles[3*triangle+1].xyz,r.meshTriangles[3*triangle+2].xyz,feature);
        float distance=dot(p-q,p-q);
        if(distance<best){second=best;best=distance;nearest=q;pseudo=r.meshNormals[7*triangle+feature].xyz;}
        else second=min(second,distance);
    }
    // Near ties retain the original BVH visit-order/feature selection.
    if(second-best<=max(1e-20,1e-5*best)){
        // BRICK_METRIC_TIE
        return meshDistanceBVH(shape,p);
    }
    // BRICK_METRIC_FAST
    float distance=sqrt(best);float side=dot(p-nearest,pseudo)<0?-1.:1.;
    float3 normal=distance>1e-12?side*(p-nearest)/distance:pseudo;
    return float4(normal,side*distance);
}
"""


def brick_shader(source, metrics=False):
    if metrics:
        from coherent_mesh_shader import instrument

        source = instrument(source)
    source = source.replace(
        "    RWStructuredBuffer<uint> meshPairMask;",
        """    RWStructuredBuffer<uint> meshPairMask;
    StructuredBuffer<uint4> brickInfo;
    StructuredBuffer<float4> brickOrigin;
    StructuredBuffer<float4> brickScale;
    StructuredBuffer<float4> brickCenters;
    StructuredBuffer<uint4> brickRanges;
    StructuredBuffer<uint> brickTriangles;""",
    )
    source = source.replace(
        "float4 meshDistance(uint shape,float3 p){",
        "float4 meshDistanceBVH(uint shape,float3 p){",
    )
    wrapper = WRAPPER
    if metrics:
        wrapper = wrapper.replace(
            "// BRICK_METRIC_LIST",
            "uint unused;InterlockedAdd(r.meshQueryMetrics[4],1,unused);InterlockedAdd(r.meshQueryMetrics[6],list.y,unused);",
        )
        wrapper = wrapper.replace(
            "// BRICK_METRIC_TIE", "InterlockedAdd(r.meshQueryMetrics[8],1,unused);"
        )
        wrapper = wrapper.replace(
            "// BRICK_METRIC_FAST", "InterlockedAdd(r.meshQueryMetrics[5],1,unused);"
        )
    at = source.index('[shader("compute")]', source.index("float4 meshDistanceBVH("))
    return source[:at] + wrapper + source[at:]
