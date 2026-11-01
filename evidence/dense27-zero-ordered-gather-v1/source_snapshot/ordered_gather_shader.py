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
