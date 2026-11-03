groupshared float4 values[256];
[shader("compute")][numthreads(256,1,1)]
void reduceRows(uint3 group:SV_GroupID,uint lane:SV_GroupIndex,uniform uint N,
                StructuredBuffer<float4> input,RWStructuredBuffer<float4> output){
    uint id=group.x*256+lane;
    for(uint row=0;row<8;row++){
        values[lane]=id<N?input[id*8+row]:float4(0);
        GroupMemoryBarrierWithGroupSync();
        for(uint step=128;step>0;step/=2){
            if(lane<step){float4 a=values[lane],b=values[lane+step];
                values[lane]=a+b;if(row==4)values[lane].w=max(a.w,b.w);}
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane==0)output[group.x*8+row]=values[0];
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")][numthreads(256,1,1)]
void reduceFabric(uint3 group:SV_GroupID,uint lane:SV_GroupIndex,uniform uint N,
                  StructuredBuffer<float4> input,RWStructuredBuffer<float4> output){
    uint id=group.x*256+lane;
    for(uint row=0;row<2;row++){
        values[lane]=id<N?input[id*2+row]:float4(0);GroupMemoryBarrierWithGroupSync();
        for(uint step=128;step>0;step/=2){if(lane<step)values[lane]+=values[lane+step];GroupMemoryBarrierWithGroupSync();}
        if(lane==0)output[group.x*2+row]=values[0];GroupMemoryBarrierWithGroupSync();
    }
}

// Per-wall rows: force/integer count, moment/power, cumulative actuator work.
groupshared uint wallCounts[256];
[shader("compute")][numthreads(256,1,1)]
void reduceWallRows(uint3 group:SV_GroupID,uint lane:SV_GroupIndex,uniform uint N,
                    uniform uint outputCount,StructuredBuffer<float4> input,
                    RWStructuredBuffer<float4> output){
    uint id=group.x*256+lane,wall=group.y;
    for(uint row=0;row<3;row++){
        float4 value=id<N?input[3*(wall*N+id)+row]:float4(0);
        if(row==0){wallCounts[lane]=asuint(value.w);value.w=0;}
        values[lane]=value;GroupMemoryBarrierWithGroupSync();
        for(uint step=128;step>0;step/=2){
            if(lane<step){values[lane]+=values[lane+step];if(row==0)wallCounts[lane]+=wallCounts[lane+step];}
            GroupMemoryBarrierWithGroupSync();
        }
        if(lane==0){float4 total=values[0];if(row==0)total.w=asfloat(wallCounts[0]);
            output[3*(wall*outputCount+group.x)+row]=total;}
        GroupMemoryBarrierWithGroupSync();
    }
}
