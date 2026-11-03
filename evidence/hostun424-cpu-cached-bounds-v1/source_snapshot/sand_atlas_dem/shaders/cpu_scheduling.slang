// CPU-only serial scheduling. Contact/geometry/integration math stays in solver.slang.
// Each case runs in its own process; these operations are never concurrent within a case.
void cpuAtomicAdd(inout uint value,uint amount,out uint old){old=value;value+=amount;}
void cpuAtomicOr(inout uint value,uint bits,out uint old){old=value;value|=bits;}
void cpuAtomicExchange(inout uint value,uint next,out uint old){old=value;value=next;}
void cpuAtomicCompareExchange(inout uint value,uint compare,uint next,out uint old){old=value;if(value==compare)value=next;}

// Preserve the local branch when high coordinates straddle a periodic boundary.
// FP32 high/residual storage is retained; only this subtraction uses FP64.
float3 centerBranch(uint i,uint j){
    double3 box=double3(r.cell[0].xyz);
    double3 d=(double3(r.position[j].xyz)-double3(r.position[i].xyz))
        +(double3(r.positionRemainder[j].xyz)-double3(r.positionRemainder[i].xyz));
    return float3(r.c.periodic!=0?d-floor(d/box+.5)*box:d);
}

// CPU replacements below are injected by name into a private source copy.
// Every candidate source node is visited once per directed pair. Pair transforms
// and conservative centre/extent bounds are shared with the Metal frontier path.
// Exact node queries, immutable keys and the overflow policy remain unchanged.
void cpuCollectMesh(uint pair){
    if(pair>=min(r.control[1],r.c.pairCapacity)||r.control[0]!=0)return;
    uint2 ij=r.pairs[pair];
    for(uint pass=0;pass<2;pass++){
        if((r.meshPairMask[pair]&(1u<<pass))==0)continue;
        uint i=pass==0?ij.x:ij.y,j=pass==0?ij.y:ij.x;
        uint4 info=r.meshClusterInfo[r.shapeId[i]];
        uint cluster=info.x;
        while(cluster<info.x+info.y){
            float3 low,high;meshPairBounds(2*pair+pass,cluster,low,high);
            bool candidate=meshBoxMayContact(r.shapeId[j],low,high);
            uint4 range=r.meshClusterRanges[cluster];
            if(candidate){
                for(uint lane=0;lane<range.y;lane++){
                    uint node=r.meshSampleIndices[range.x+lane];float4 pd,na;
                    if(nodeContact(i,j,node,pd,na)){
                        uint offset=r.control[2]++;
                        if(offset>=r.c.contactCapacity){fail(32);return;}
                        r.meta[offset]=uint4(i,j,node,0);r.pointDepth[offset]=pd;r.normalArea[offset]=na;
                        r.contactCounts[pair]++;
                    }
                }
            }
            cluster=candidate?cluster+1:range.z;
        }
    }
}

// BEGIN collectMeshContacts
[shader("compute")][numthreads(1,1,1)]
void collectMeshContacts(uint id:SV_DispatchThreadID){cpuCollectMesh(id);}
// END collectMeshContacts
// BEGIN collectMeshFrontiers
[shader("compute")][numthreads(1,1,1)]
void collectMeshFrontiers(uint id:SV_DispatchThreadID){cpuCollectMesh(id);}
// END collectMeshFrontiers

// BEGIN countContacts
[shader("compute")][numthreads(1,1,1)]
void countContacts(uint pair:SV_DispatchThreadID){
    if(pair>=r.c.pairCapacity)return;
    uint count=0;
    if(pair<r.control[1]&&r.control[0]==0){
        uint2 ij=r.pairs[pair];uint a=r.sampleInfo[r.shapeId[ij.x]].y;
        uint total=a+r.sampleInfo[r.shapeId[ij.y]].y;
        for(uint s=0;s<total;s++){
            float4 pd,na;
            if(nodeContact(s<a?ij.x:ij.y,s<a?ij.y:ij.x,s<a?s:s-a,pd,na))count++;
        }
    }
    r.contactCounts[pair]=count;
}
// END countContacts
// BEGIN emitContacts
[shader("compute")][numthreads(1,1,1)]
void emitContacts(uint pair:SV_DispatchThreadID){
    if(pair>=min(r.control[1],r.c.pairCapacity)||r.control[0]!=0)return;
    uint2 ij=r.pairs[pair];uint a=r.sampleInfo[r.shapeId[ij.x]].y;
    uint total=a+r.sampleInfo[r.shapeId[ij.y]].y,offset=r.contactOffsets[pair];
    for(uint s=0;s<total;s++){
        uint i=s<a?ij.x:ij.y,j=s<a?ij.y:ij.x,node=s<a?s:s-a;float4 pd,na;
        if(nodeContact(i,j,node,pd,na)){
            if(offset>=r.c.contactCapacity){fail(32);return;}
            r.meta[offset]=uint4(i,j,node,0);r.pointDepth[offset]=pd;r.normalArea[offset]=na;offset++;
        }
    }
}
// END emitContacts
// BEGIN collectWallContacts
[shader("compute")][numthreads(1,1,1)]
void collectWallContacts(uint id:SV_DispatchThreadID){
    uint body=id%r.c.N,wall=id/r.c.N;
    if(wall>=r.c.wallCount||r.control[0]!=0)return;
    float4 plane=r.wallPlane[wall];float2 centreDistance=wallCentreDistance(body,wall);
    float phi=centreDistance.x+centreDistance.y;
    float pad=64.*1.1920928955078125e-7*(1.+length(r.position[body].xyz)+abs(plane.w));
    if(phi>r.body[body].z+pad)return;
    if(r.c.meshGeometry==0){
        uint count=r.sampleInfo[r.shapeId[body]].y;
        for(uint node=0;node<count;node++)appendWallContact(body,wall,node);
        return;
    }
    uint4 info=r.meshClusterInfo[r.shapeId[body]];uint cluster=info.x;
    while(cluster<info.x+info.y){
        bool candidate=wallClusterMayContact(body,wall,cluster);uint4 range=r.meshClusterRanges[cluster];
        if(candidate)for(uint lane=0;lane<range.y;lane++)appendWallContact(body,wall,r.meshSampleIndices[range.x+lane]);
        cluster=candidate?cluster+1:range.z;
    }
}
// END collectWallContacts
