"""Experimental contiguous adjacency and compensated workgroup gather source builder."""


def precision(source):
    source = source.replace(
        "float best=1e30;float3 nearest=0,pseudo=0;",
        "float best=1e30;float3 nearest=0,pseudo=0;uint nearestFeature=0;",
    )
    source = source.replace(
        "best=distance;nearest=q;pseudo=",
        "best=distance;nearest=q;nearestFeature=feature;pseudo=",
    )
    return source.replace(
        "normal=distance>1e-12?", "normal=distance>1e-12&&nearestFeature!=0?"
    )


def segmented(source):
    source = source.replace(
        "    Config c;",
        "    Config c;\n    RWStructuredBuffer<uint> adjCounts,adjOffsets,adjCursor,adjEntries;",
    )
    source = source.replace(
        "if(id<r.c.N)r.heads[id]=EMPTY;",
        "if(id<r.c.N){r.heads[id]=EMPTY;r.adjCounts[id]=0;r.adjCursor[id]=0;}",
    )
    source = source.replace(
        "r.next[2*id]=oldHead;r.next[2*id+1]=EMPTY;",
        "r.next[2*id]=oldHead;r.next[2*id+1]=EMPTY;uint unused;InterlockedAdd(r.adjCounts[i],1,unused);",
    )
    old = "uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);r.next[2*id]=oldHead;\n    InterlockedExchange(r.heads[j],2*id+1,oldHead);r.next[2*id+1]=oldHead;"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "uint unused;InterlockedAdd(r.adjCounts[i],1,unused);InterlockedAdd(r.adjCounts[j],1,unused);",
    )
    start = source.index('[shader("compute")][numthreads(64,1,1)]\nvoid gather(')
    end = source.index("// Separate dispatch", start)
    gather = source[start:end]
    gather = gather.replace(
        "void gather(uint id:SV_DispatchThreadID)",
        "void gather(uint id:SV_GroupID,uint lane:SV_GroupThreadID)",
    )
    gather = gather.replace(
        "uint entry=r.heads[id];uint visited=0;\n    while(entry!=EMPTY){\n        if(visited++>2*r.c.contactCapacity){fail(128);return;}",
        "for(uint index=lane;index<r.adjCounts[id];index+=64){\n        uint entry=r.adjEntries[r.adjOffsets[id]+index];",
    )
    gather = gather.replace("entry=r.next[entry];continue;", "continue;").replace(
        "        entry=r.next[entry];\n", ""
    )
    fields = {
        "F": "float4(f,maxDepth)",
        "FL": "float4(fLow,0)",
        "T": "float4(t,dissipation)",
        "TL": "float4(tLow,0)",
        "S0": "float4(stress0,stiffnessRate)",
        "S1": "float4(stress1,viscousRate)",
        "S2": "float4(stress2,normalPower)",
        "A": "float4(energy,denom,0,0)",
    }
    reduction = "\n".join(f"    shared{k}[lane]={v};" for k, v in fields.items())
    reduction += "\n    GroupMemoryBarrierWithGroupSync();\n    for(uint stride=32;stride>0;stride/=2){\n        if(lane<stride){\n            float3 hf=sharedF[lane].xyz,lf=sharedFL[lane].xyz,ht=sharedT[lane].xyz,lt=sharedTL[lane].xyz;\n            compensated3(hf,lf,sharedF[lane+stride].xyz);compensated3(hf,lf,sharedFL[lane+stride].xyz);\n            compensated3(ht,lt,sharedT[lane+stride].xyz);compensated3(ht,lt,sharedTL[lane+stride].xyz);\n            sharedF[lane]=float4(hf,max(sharedF[lane].w,sharedF[lane+stride].w));sharedFL[lane]=float4(lf,0);\n            sharedT[lane]=float4(ht,sharedT[lane].w+sharedT[lane+stride].w);sharedTL[lane]=float4(lt,0);\n"
    reduction += "\n".join(
        f"            shared{k}[lane]+=shared{k}[lane+stride];"
        for k in ["S0", "S1", "S2", "A"]
    )
    reduction += "\n        }\n        GroupMemoryBarrierWithGroupSync();\n    }\n    if(lane!=0)return;\n    f=sharedF[0].xyz;fLow=sharedFL[0].xyz;t=sharedT[0].xyz;tLow=sharedTL[0].xyz;\n    maxDepth=sharedF[0].w;dissipation=sharedT[0].w;\n    stress0=sharedS0[0].xyz;stress1=sharedS1[0].xyz;stress2=sharedS2[0].xyz;\n    stiffnessRate=sharedS0[0].w;viscousRate=sharedS1[0].w;normalPower=sharedS2[0].w;\n    energy=sharedA[0].x;denom=sharedA[0].y;\n"
    gather = gather.replace(
        "    f=rounded3(f+fLow);", reduction + "    f=rounded3(f+fLow);"
    )
    declarations = (
        "\n".join(f"groupshared float4 shared{k}[64];" for k in fields) + "\n"
    )
    scatter = """
[shader("compute")][numthreads(64,1,1)]
void scatterAdjacency(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint4 key=r.meta[id];uint slot;
    InterlockedAdd(r.adjCursor[key.x],1,slot);r.adjEntries[r.adjOffsets[key.x]+slot]=2*id;
    if(key.y<r.c.N){InterlockedAdd(r.adjCursor[key.y],1,slot);r.adjEntries[r.adjOffsets[key.y]+slot]=2*id+1;}
}
"""
    return source[:start] + scatter + declarations + gather + source[end:]
