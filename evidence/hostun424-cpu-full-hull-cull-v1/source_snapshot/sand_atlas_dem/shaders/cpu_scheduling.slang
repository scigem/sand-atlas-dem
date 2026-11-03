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

// CPU conservative rejection uses all existing outward-rounded hull planes.
bool meshMayContain(uint shape,float3 p){
    uint root=r.meshRoots[shape].x;
    float3 low=r.meshMin[root].xyz,high=r.meshMax[root].xyz;
    if(any(p<low)||any(p>high))return false;
    float3 direction=(p-.5*(low+high))/(high-low);
    float3 a=abs(direction);uint axis=a.x>=a.y&&a.x>=a.z?0:(a.y>=a.z?1:2);
    uint group=2*axis+(direction[axis]>=0?1:0);
    uint4 info=r.meshHullInfo[6*shape+group];
    for(uint i=0;i<info.y;i++){
        float4 plane=r.meshPlanes[info.x+i];
        if(dot(plane.xyz,p)+plane.w>0)return false;
    }
    for(uint other=0;other<6;other++){
        if(other==group)continue;
        uint4 extra=r.meshHullInfo[6*shape+other];
        for(uint k=0;k<extra.y;k++){
            float4 plane=r.meshPlanes[extra.x+k];
            if(dot(plane.xyz,p)+plane.w>0)return false;
        }
    }
    return true;
}

bool meshBoxMayContact(uint shape,float3 low,float3 high){
    uint root=r.meshRoots[shape].x;
    float3 targetLow=r.meshMin[root].xyz,targetHigh=r.meshMax[root].xyz;
    if(any(high<targetLow)||any(low>targetHigh))return false;
    float3 center=.5*(low+high),half=.5*(high-low);
    float3 direction=(center-.5*(targetLow+targetHigh))/(targetHigh-targetLow);
    float3 a=abs(direction);uint axis=a.x>=a.y&&a.x>=a.z?0:(a.y>=a.z?1:2);
    uint group=2*axis+(direction[axis]>=0?1:0);
    uint4 info=r.meshHullInfo[6*shape+group];
    for(uint i=0;i<info.y;i++){
        float4 plane=r.meshPlanes[info.x+i];
        if(dot(plane.xyz,center)+plane.w-dot(abs(plane.xyz),half)>0)return false;
    }
    for(uint other=0;other<6;other++){
        if(other==group)continue;
        uint4 extra=r.meshHullInfo[6*shape+other];
        for(uint k=0;k<extra.y;k++){
            float4 plane=r.meshPlanes[extra.x+k];
            if(dot(plane.xyz,center)+plane.w-dot(abs(plane.xyz),half)>0)return false;
        }
    }
    return true;
}

// CPU query: retain pushed FP32 box bounds and defer normal fetch to final nearest feature.
float4 meshDistance(uint shape,float3 p,bool surfaceNormal=false){
    uint root=r.meshRoots[shape].x;
    uint pending[32];float pendingDistance[32];uint top=1;pending[0]=root;
    pendingDistance[0]=meshBoxDistance(root,p);
    float best=1e30;float3 nearest=0;uint nearestNormal=0;bool found=false;
    while(top>0){
        uint slot=--top,node=pending[slot];
        if(pendingDistance[slot]>best)continue;
        uint4 link=r.meshLinks[node];
        if(link.z==0){
            uint a=node+1,b=r.meshLinks[a].x;
            float da=meshBoxDistance(a,p),db=meshBoxDistance(b,p);
            if(da<db){uint swap=a;a=b;b=swap;float ds=da;da=db;db=ds;}
            // Far child first; LIFO visits the nearer child next.
            if(top+2>32){fail(1024);return float4(0,0,0,1e20);}
            if(da<=best){pending[top]=a;pendingDistance[top++]=da;}
            if(db<=best){pending[top]=b;pendingDistance[top++]=db;}
            continue;
        }
        for(uint j=0;j<link.z;j++){
            uint triangle=link.y+j,feature;
            float3 q=closestTriangle(p,r.meshTriangles[3*triangle].xyz,r.meshTriangles[3*triangle+1].xyz,r.meshTriangles[3*triangle+2].xyz,feature);
            float distance=dot(p-q,p-q);
            if(distance<best){best=distance;nearest=q;nearestNormal=7*triangle+feature;found=true;}
        }
    }
    float3 pseudo=found?r.meshNormals[nearestNormal].xyz:float3(0);
    float distance=sqrt(best);float side=dot(p-nearest,pseudo)<0?-1.:1.;
    float3 normal=surfaceNormal?pseudo:(distance>1e-12?side*(p-nearest)/distance:pseudo);
    return float4(normal,side*distance);
}

// CPU replacements below are injected by name into a private source copy.
// Every source node is visited once per directed pair. The existing conservative
// cluster rejection, exact query, immutable key and overflow policy are retained.
void cpuCollectMesh(uint pair){
    if(pair>=min(r.control[1],r.c.pairCapacity)||r.control[0]!=0)return;
    uint2 ij=r.pairs[pair];
    for(uint pass=0;pass<2;pass++){
        uint i=pass==0?ij.x:ij.y,j=pass==0?ij.y:ij.x;
        uint4 info=r.meshClusterInfo[r.shapeId[i]];
        uint cluster=info.x;
        while(cluster<info.x+info.y){
            bool candidate=meshClusterMayContact(i,j,cluster);
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
