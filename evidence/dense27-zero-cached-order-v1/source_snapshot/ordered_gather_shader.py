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
        "groupshared uint sortedEndpoints[8192];\ngroupshared uint cacheMiss;",
    )
    old = "    uint offset=r.adjOffsets[body];\n    for(uint i=lane;i<size;i+=64)"
    new = r"""    uint offset=r.adjOffsets[body];
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
            sortedEndpoints[i]=found==EMPTY?EMPTY:2*found+cached.w;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(cacheMiss==0){
        for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=sortedEndpoints[i];
        if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[0],1,unused);}
        return;
    }
    if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[1],1,unused);}
    for(uint i=lane;i<size;i+=64)"""
    assert old in source
    source = source.replace(old, new)
    old = "    for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=sortedEndpoints[i];\n}"
    new = r"""    for(uint i=lane;i<count;i+=64){
        uint entry=sortedEndpoints[i];r.adjEntries[offset+i]=entry;
        r.adjCacheKeys[8192*body+i]=uint4(r.meta[entry/2].xyz,entry&1);
    }
    if(lane==0)r.adjCacheCounts[body]=count;
}"""
    assert old in source
    return source.replace(old, new)
