// Hierarchical exclusive uint scan; works without subgroup-size assumptions.
groupshared uint scratch[256];
[shader("compute")]
[numthreads(256,1,1)]
void blockScan(uint3 group: SV_GroupID,uint lane: SV_GroupIndex,
               uniform uint N, StructuredBuffer<uint> input,
               RWStructuredBuffer<uint> output, RWStructuredBuffer<uint> totals)
{
    uint id=group.x*256+lane;
    uint v=id<N?input[id]:0;
    scratch[lane]=v;
    GroupMemoryBarrierWithGroupSync();
    for(uint shift=1;shift<256;shift*=2){
        uint a=lane>=shift?scratch[lane-shift]:0;
        GroupMemoryBarrierWithGroupSync();
        scratch[lane]+=a;
        GroupMemoryBarrierWithGroupSync();
    }
    if(id<N) output[id]=scratch[lane]-v;
    if(lane==255) totals[group.x]=scratch[255];
}
[shader("compute")]
[numthreads(256,1,1)]
void addBlocks(uint id: SV_DispatchThreadID,uniform uint N,
               RWStructuredBuffer<uint> output,StructuredBuffer<uint> offsets)
{
    if(id<N)output[id]+=offsets[id/256];
}
