"""Experimental fixed canonical endpoint slots with exact GPU membership validation."""

from ordered_gather_shader import ordered


def persistent(source):
    result = ordered(source)
    result = result.replace(
        "    Config c;",
        "    Config c;\n    RWStructuredBuffer<uint4> adjCacheKeys;\n    RWStructuredBuffer<uint> adjCacheCounts,adjCacheMetrics,slotNeedsBuild;",
    )
    result = result.replace("r.adjOffsets[id]+index", "8192*id+index")
    result = result.replace("r.adjOffsets[body]", "8192*body")
    start = result.index("void scatterAdjacency(")
    end = result.index("groupshared float4 sharedF", start)
    old = result[start:end]
    new = old.replace(
        "    InterlockedAdd(r.adjCursor[key.x],1,slot);r.adjEntries[r.adjOffsets[key.x]+slot]=2*id;",
        "    if(r.slotNeedsBuild[key.x]!=0){InterlockedAdd(r.adjCursor[key.x],1,slot);r.adjEntries[8192*key.x+slot]=2*id;}",
    )
    new = new.replace(
        "if(key.y<r.c.N){InterlockedAdd(r.adjCursor[key.y],1,slot);r.adjEntries[r.adjOffsets[key.y]+slot]=2*id+1;}",
        "if(key.y<r.c.N&&r.slotNeedsBuild[key.y]!=0){InterlockedAdd(r.adjCursor[key.y],1,slot);r.adjEntries[8192*key.y+slot]=2*id+1;}",
    )
    assert new != old
    result = result[:start] + new + result[end:]
    result = result.replace(
        "    uint count=r.adjCounts[body];",
        "    if(r.slotNeedsBuild[body]==0)return;\n    uint count=r.adjCounts[body];",
    )
    old = "    for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=sortedEndpoints[i];\n}"
    new = """    for(uint i=lane;i<count;i+=64){
        uint entry=sortedEndpoints[i];r.adjEntries[offset+i]=entry;
        r.adjCacheKeys[8192*body+i]=uint4(r.meta[entry/2].xyz,entry&1);
    }
    if(lane==0)r.adjCacheCounts[body]=count;
}"""
    assert old in result
    result = result.replace(old, new)
    return (
        result
        + r"""
// Separate entry point: only a single shared uint, independent of sort scratch.
groupshared uint slotMiss;
[shader("compute")][numthreads(64,1,1)]
void validateSlots(uint body:SV_GroupID,uint lane:SV_GroupThreadID){
    if(body>=r.c.N||r.control[0]!=0)return;
    uint count=r.adjCounts[body];
    if(count>8192){if(lane==0)fail(16384);return;}
    if(lane==0)slotMiss=r.adjCacheCounts[body]==count?0:1;
    GroupMemoryBarrierWithGroupSync();
    if(slotMiss==0){
        for(uint i=lane;i<count;i+=64){
            uint4 cached=r.adjCacheKeys[8192*body+i];
            uint slot=hashKey(cached),found=EMPTY;
            for(uint probe=0;probe<r.c.hashSize;probe++){
                uint current=r.hash[slot];if(current==EMPTY)break;
                if(keyEqual(cached,r.meta[current])){found=current;break;}
                slot=(slot+1)&(r.c.hashSize-1);
            }
            if(found==EMPTY){uint unused;InterlockedOr(slotMiss,1,unused);}
            r.adjEntries[8192*body+i]=found==EMPTY?EMPTY:2*found+cached.w;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(lane==0){
        r.slotNeedsBuild[body]=slotMiss;
        uint unused;InterlockedAdd(r.adjCacheMetrics[slotMiss==0?0:1],1,unused);
    }
}
"""
    )
