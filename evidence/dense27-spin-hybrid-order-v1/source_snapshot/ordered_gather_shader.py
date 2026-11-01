"""Bounded experimental canonical GPU endpoint sorting; never truncates contacts."""

from segmented_shader import segmented


def ordered(source):
    source = segmented(source)
    return (
        source
        + r"""
// Prototype limit is checked before any load; failure aborts the trial.
// This is scratch for this entry point only, not the gather entry point.
groupshared uint sortedEndpoints[8192];
bool endpointLess(uint a,uint b){
    if(a==EMPTY)return false;
    if(b==EMPTY)return true;
    uint4 ka=r.meta[a/2],kb=r.meta[b/2];
    if(ka.x!=kb.x)return ka.x<kb.x;
    if(ka.y!=kb.y)return ka.y<kb.y;
    if(ka.z!=kb.z)return ka.z<kb.z;
    if(ka.w!=kb.w)return ka.w<kb.w;
    return (a&1)<(b&1);
}
[shader("compute")][numthreads(64,1,1)]
void orderAdjacency(uint body:SV_GroupID,uint lane:SV_GroupThreadID){
    if(body>=r.c.N||r.control[0]!=0)return;
    uint count=r.adjCounts[body];
    if(count>8192){if(lane==0)fail(16384);return;}
    uint size=1;while(size<count)size*=2;
    uint offset=r.adjOffsets[body];
    for(uint i=lane;i<size;i+=64)sortedEndpoints[i]=i<count?r.adjEntries[offset+i]:EMPTY;
    GroupMemoryBarrierWithGroupSync();
    for(uint width=2;width<=size;width*=2){
        for(uint stride=width/2;stride>0;stride/=2){
            for(uint i=lane;i<size;i+=64){
                uint j=i^stride;
                if(j>i){
                    uint a=sortedEndpoints[i],b=sortedEndpoints[j];
                    bool ascending=(i&width)==0;
                    if(ascending?endpointLess(b,a):endpointLess(a,b)){
                        sortedEndpoints[i]=b;sortedEndpoints[j]=a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=sortedEndpoints[i];
}
"""
    )


def cached_ordered(source):
    source = ordered(source)
    source = source.replace(
        "    Config c;",
        "    Config c;\n    RWStructuredBuffer<uint4> adjCacheKeys;\n    RWStructuredBuffer<uint> adjCacheCounts,adjCacheMetrics;",
    )
    source = source.replace(
        "groupshared uint sortedEndpoints[8192];",
        "groupshared uint sortedEndpoints[8192];",
    )
    old = "    uint offset=r.adjOffsets[body];\n    for(uint i=lane;i<size;i+=64)"
    new = r"""    uint offset=r.adjOffsets[body];uint lastMapped=EMPTY;
    if(lane==0)cacheMiss=r.adjCacheCounts[body]==count?0:1;
    GroupMemoryBarrierWithGroupSync();
    if(cacheMiss==0){
        for(uint i=lane;i<count;i+=64){
            uint4 cached=r.adjCacheKeys[8192*body+i];
            uint slot=hashKey(cached);uint found=EMPTY;
            for(uint probe=0;probe<r.c.hashSize;probe++){
                uint current=r.hash[slot];if(current==EMPTY)break;
                if(keyEqual(cached,r.meta[current])){found=current;break;}
                slot=(slot+1)&(r.c.hashSize-1);
            }
            if(found==EMPTY){uint unused;InterlockedOr(cacheMiss,1,unused);}
            uint mapped=found==EMPTY?EMPTY:2*found+cached.w;
            if(i==8191)lastMapped=mapped;else sortedEndpoints[i]=mapped;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    bool reuse=cacheMiss==0;
    GroupMemoryBarrierWithGroupSync();
    if(reuse){
        for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=i==8191?lastMapped:sortedEndpoints[i];
        if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[0],1,unused);}
        return;
    }
    if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[1],1,unused);}
    for(uint i=lane;i<size;i+=64)"""
    assert old in source
    source = source.replace(old, new.replace("cacheMiss", "sortedEndpoints[8191]"))
    old = "    for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=sortedEndpoints[i];\n}"
    new = r"""    for(uint i=lane;i<count;i+=64){
        uint entry=sortedEndpoints[i];r.adjEntries[offset+i]=entry;
        r.adjCacheKeys[8192*body+i]=uint4(r.meta[entry/2].xyz,entry&1);
    }
    if(lane==0)r.adjCacheCounts[body]=count;
}"""
    assert old in source
    return source.replace(old, new)


def hybrid(source):
    result = cached_ordered(source)
    result = result.replace(
        "ParameterBlock<Resources> r;",
        "ParameterBlock<Resources> r;\nbool denseGather(){return r.control[2]>=64*r.c.N;}",
    )
    old = "uint unused;InterlockedAdd(r.adjCounts[i],1,unused);InterlockedAdd(r.adjCounts[j],1,unused);"
    new = """if(denseGather()){
        uint unused;InterlockedAdd(r.adjCounts[i],1,unused);InterlockedAdd(r.adjCounts[j],1,unused);
    }else{
        uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);r.next[2*id]=oldHead;
        InterlockedExchange(r.heads[j],2*id+1,oldHead);r.next[2*id+1]=oldHead;
    }"""
    assert old in result
    result = result.replace(old, new)
    for name, args in [
        ("scatterAdjacency", "uint id:SV_DispatchThreadID"),
        ("orderAdjacency", "uint body:SV_GroupID,uint lane:SV_GroupThreadID"),
    ]:
        old = f"void {name}({args}){{"
        assert old in result
        result = result.replace(old, old + "\n    if(!denseGather())return;")
    old = '[shader("compute")][numthreads(64,1,1)]\nvoid gather(uint id:SV_GroupID,uint lane:SV_GroupThreadID)'
    assert old in result
    result = result.replace(old, "void gatherDense(uint id,uint lane)")
    start = source.index('[shader("compute")][numthreads(64,1,1)]\nvoid gather(')
    end = source.index("// Separate dispatch", start)
    sparse = source[start:end].replace(
        '[shader("compute")][numthreads(64,1,1)]\nvoid gather(uint id:SV_DispatchThreadID)',
        "void gatherSparse(uint id)",
    )
    return (
        result
        + sparse
        + r"""
[shader("compute")][numthreads(64,1,1)]
void gather(uint id:SV_DispatchThreadID,uint group:SV_GroupID,uint lane:SV_GroupThreadID){
    if(denseGather())gatherDense(group,lane);
    else gatherSparse(id);
}
"""
    )
