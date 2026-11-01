"""Exact BVH query with a per-lane previous-triangle upper bound; experiment only."""


def coherent(source):
    start = source.index("float4 meshDistance(uint shape,float3 p){")
    end = source.index('[shader("compute")]', start)
    original = source[start:end]
    cached = original.replace(
        "float4 meshDistance(uint shape,float3 p){",
        "float4 meshDistanceCoherent(uint shape,float3 p,inout uint seed){",
    )
    cached = cached.replace(
        "    while(top>0){",
        """    // Seed supplies an upper bound only. Never reuse an old point/normal.
    // Padding ensures the seed itself can be discovered in the ordinary walk.
    if(seed!=0xffffffffu){
        uint feature;
        float3 q=closestTriangle(p,r.meshTriangles[3*seed].xyz,r.meshTriangles[3*seed+1].xyz,r.meshTriangles[3*seed+2].xyz,feature);
        float d=dot(p-q,p-q);
        best=d+max(1e-12,abs(d)*1e-5);
    }
    while(top>0){""",
    )
    cached = cached.replace(
        "if(distance<best){best=distance;",
        "if(distance<best){seed=triangle;best=distance;",
    )
    source = source[:end] + cached + source[end:]
    start = source.index("bool nodeContact(")
    end = source.index('[shader("compute")]', start)
    helper = (
        source[start:end]
        .replace(
            "bool nodeContact(uint src,uint dst,uint node,out float4 pd,out float4 na){",
            "bool nodeContactCoherent(uint src,uint dst,uint node,out float4 pd,out float4 na,inout uint seed){",
        )
        .replace(
            "f=meshDistance(targetShape,local);",
            "f=meshDistanceCoherent(targetShape,local,seed);",
        )
    )
    source = source[:end] + helper + source[end:]
    start = source.index("void collectMeshSubtree(")
    end = source.index("// Metal currently lacks", start)
    part = source[start:end].replace(
        "    uint cluster=subtree.x;",
        "    uint seed=0xffffffffu;\n    uint cluster=subtree.x;",
    )
    part = part.replace(
        "if(nodeContact(i,j,node,pd,na)){",
        "if(nodeContactCoherent(i,j,node,pd,na,seed)){",
    )
    return source[:start] + part + source[end:]


def instrument(source):
    source = source.replace(
        "    RWStructuredBuffer<uint> meshPairMask;",
        "    RWStructuredBuffer<uint> meshPairMask;\n    RWStructuredBuffer<uint> meshQueryMetrics;",
    )
    for name in ["meshDistance", "meshDistanceCoherent"]:
        marker = "float4 " + name + "("
        if marker not in source:
            continue
        start = source.index(marker)
        end = source.index("\n}", start) + 2
        part = source[start:end]
        part = part.replace(
            "    uint root=", "    uint visits=0,triangles=0;\n    uint root="
        )
        part = part.replace(
            "uint node=pending[--top];", "uint node=pending[--top];visits++;"
        )
        part = part.replace(
            "uint triangle=link.y+j,feature;",
            "uint triangle=link.y+j,feature;triangles++;",
        )
        part = part.replace(
            "    float distance=sqrt(best);",
            """    uint unused;
    InterlockedAdd(r.meshQueryMetrics[0],1,unused);
    InterlockedAdd(r.meshQueryMetrics[1],visits,unused);
    InterlockedAdd(r.meshQueryMetrics[2],triangles,unused);
    float distance=sqrt(best);""",
        )
        if name.endswith("Coherent"):
            part = part.replace(
                "    if(seed!=0xffffffffu){",
                "    if(seed!=0xffffffffu){uint unused;InterlockedAdd(r.meshQueryMetrics[3],1,unused);",
            )
        source = source[:start] + part + source[end:]
    return source
