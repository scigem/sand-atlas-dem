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


def fused(source):
    result = persistent(source)
    marker = '[shader("compute")][numthreads(64,1,1)]\nvoid resolveContacts('
    start = source.index(marker)
    end = source.index('[shader("compute")]', start + len(marker))
    original = source[start:end]
    evaluation = original.replace(marker, "ContactEvaluation evaluateContact(").replace(
        "uint id:SV_DispatchThreadID", "uint id"
    )
    guard = "    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;"
    assert guard in evaluation
    evaluation = evaluation.replace(
        guard,
        "    ContactEvaluation value;value.slip=0;value.force=0;value.normalLoss=0;value.sliding=false;",
    )
    wall_append = """        uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);
        r.next[2*id]=oldHead;r.next[2*id+1]=EMPTY;
"""
    assert wall_append in evaluation
    evaluation = evaluation.replace(wall_append, "")
    append = """    uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);r.next[2*id]=oldHead;
    InterlockedExchange(r.heads[j],2*id+1,oldHead);r.next[2*id+1]=oldHead;
"""
    assert append in evaluation
    evaluation = evaluation.replace(append, "    return value;\n")
    evaluation = (
        evaluation.replace("r.slip[id]", "value.slip")
        .replace("r.contactForce[id]", "value.force")
        .replace("r.normalLossPower[id]", "value.normalLoss")
    )
    counter = "uint unused;InterlockedAdd(r.control[6],1,unused);"
    assert evaluation.count(counter) == 2
    evaluation = evaluation.replace(counter, "value.sliding=true;")
    evaluation = evaluation.replace("return;", "return value;")
    declaration = "struct ContactEvaluation {float4 slip;float4 force;float normalLoss;bool sliding;};\n"
    # Replace force resolution with endpoint counting only. Wall heads remain for diagnostics.
    start = result.index(marker)
    end = result.index('[shader("compute")]', start + len(marker))
    counting = r"""[shader("compute")][numthreads(64,1,1)]
void resolveContacts(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint4 key=r.meta[id];uint unused;InterlockedAdd(r.adjCounts[key.x],1,unused);
    if(key.y<r.c.N){InterlockedAdd(r.adjCounts[key.y],1,unused);}
    else{
        uint oldHead;InterlockedExchange(r.heads[key.x],2*id,oldHead);
        r.next[2*id]=oldHead;r.next[2*id+1]=EMPTY;
    }
}
"""
    result = result[:start] + declaration + evaluation + counting + result[end:]
    start = result.index("void gather(")
    end = result.index("// Separate dispatch", start)
    gather = result[start:end]
    old = "        float3 force=r.contactForce[contact].xyz;"
    assert old in gather
    new = """        ContactEvaluation contactValue=evaluateContact(contact);
        if(source){
            r.slip[contact]=contactValue.slip;r.contactForce[contact]=contactValue.force;
            r.normalLossPower[contact]=contactValue.normalLoss;
            if(contactValue.sliding){uint unused;InterlockedAdd(r.control[6],1,unused);}
        }
        float3 force=contactValue.force.xyz;"""
    gather = gather.replace(old, new)
    gather = (
        gather.replace("r.contactForce[contact].w", "contactValue.force.w")
        .replace("r.slip[contact].w", "contactValue.slip.w")
        .replace("r.normalLossPower[contact]", "contactValue.normalLoss")
    )
    # Restore the single-owner write transformed by the diagnostic substitution.
    gather = gather.replace(
        "contactValue.normalLoss=contactValue.normalLoss;",
        "r.normalLossPower[contact]=contactValue.normalLoss;",
    )
    return result[:start] + gather + result[end:]
