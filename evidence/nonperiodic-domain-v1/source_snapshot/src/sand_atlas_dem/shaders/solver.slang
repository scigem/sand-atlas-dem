// Standalone Slang kernels. Quaternion order xyzw; body-to-world rotations.
[shader("compute")]
[numthreads(64,1,1)]
void smoke(uint tid: SV_DispatchThreadID, uniform uint N,
           StructuredBuffer<float> a, RWStructuredBuffer<float> b)
{
    if (tid < N) b[tid] = a[tid]*a[tid]+1.0;
}

struct Config {
    uint N; uint shapes; uint pairCapacity; uint contactCapacity;
    uint binCapacity; uint binCount; uint3 bins; uint hashSize;
    float dt; float kn; float kt; float mu;
    float skin; float drag; float maxOverlap; uint historyStep;
    uint meshGeometry; uint periodic;
};
struct Resources {
    Config c;
    RWStructuredBuffer<uint> control;
    RWStructuredBuffer<float4> cell; // lengths, imposed diagonal log rates, initial lengths
    RWStructuredBuffer<float4> position;
    RWStructuredBuffer<float4> positionRemainder;
    RWStructuredBuffer<float4> velocityRemainder;
    RWStructuredBuffer<float4> momentumRemainder;
    RWStructuredBuffer<float4> rotationRemainder;
    RWStructuredBuffer<float4> referencePositionRemainder;
    RWStructuredBuffer<float4> cellLog; // accumulated diagonal log strain, residual
    RWStructuredBuffer<float4> accumulatedRemainder;
    RWStructuredBuffer<float4> velocity;
    RWStructuredBuffer<float4> rotation;
    RWStructuredBuffer<float4> momentum;
    RWStructuredBuffer<float4> force;
    RWStructuredBuffer<float4> torque;
    RWStructuredBuffer<float4> referencePosition;
    RWStructuredBuffer<float4> referenceCell;
    StructuredBuffer<uint> shapeId;
    StructuredBuffer<float4> body; // scale,mass,bounding radius, equivalent diameter
    StructuredBuffer<float4> invI; // three rows per particle in body coordinates
    StructuredBuffer<uint4> grids;
    StructuredBuffer<uint4> sampleInfo;
    StructuredBuffer<float4> gridOrigin;
    StructuredBuffer<float> distance;
    StructuredBuffer<float4> sample;
    StructuredBuffer<uint4> meshRoots;
    StructuredBuffer<uint4> meshHullInfo;
    StructuredBuffer<float4> meshPlanes;
    StructuredBuffer<float4> meshMin;
    StructuredBuffer<float4> meshMax;
    StructuredBuffer<uint4> meshLinks;
    StructuredBuffer<float4> meshTriangles;
    StructuredBuffer<float4> meshNormals;
    StructuredBuffer<uint4> meshClusterInfo;
    StructuredBuffer<uint4> meshClusterRanges;
    StructuredBuffer<float4> meshClusterMin;
    StructuredBuffer<float4> meshClusterMax;
    StructuredBuffer<uint> meshSampleIndices;
    RWStructuredBuffer<uint> binCounts;
    RWStructuredBuffer<uint> binParticles;
    RWStructuredBuffer<uint> pairCounts;
    RWStructuredBuffer<uint> pairOffsets;
    RWStructuredBuffer<uint2> pairs;
    RWStructuredBuffer<uint> contactCounts;
    RWStructuredBuffer<uint> contactOffsets;
    RWStructuredBuffer<uint4> meta;
    RWStructuredBuffer<float4> pointDepth;
    RWStructuredBuffer<float4> normalArea;
    RWStructuredBuffer<float4> slip;
    RWStructuredBuffer<float4> contactForce;
    StructuredBuffer<uint4> oldMeta;
    StructuredBuffer<float4> oldSlip;
    StructuredBuffer<float4> oldNormal;
    StructuredBuffer<uint> oldHash;
    RWStructuredBuffer<uint> hash;
    RWStructuredBuffer<uint> heads;
    RWStructuredBuffer<uint> next;
    RWStructuredBuffer<float4> particleStress; // 3 rows per particle, contact contribution
    RWStructuredBuffer<float4> particleDiagnostic; // kinetic, elastic proxy, force norm, contact force norms
    RWStructuredBuffer<float4> diagnosticRows;
    RWStructuredBuffer<float4> accumulated; // friction loss, drag loss, cell work, reserved
    RWStructuredBuffer<float4> fabricRows;
};
ParameterBlock<Resources> r;
static const uint EMPTY=0xffffffff;

float3 qrot(float4 q,float3 v){return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
float3 qinv(float4 q,float3 v){return qrot(float4(-q.xyz,q.w),v);}
float4 qmul(float4 a,float4 b){return float4(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));}
// control[15] is an immutable runtime zero. Bitwise materialization prevents
// Metal fast-math reassociation from algebraically deleting the residual.
float3 rounded3(float3 x){return asfloat(asuint(x)^r.control[15]);}
float4 rounded4(float4 x){return asfloat(asuint(x)^r.control[15]);}
void compensated3(inout float3 high,inout float3 low,float3 increment){
    float3 y=rounded3(increment+low);float3 next=rounded3(high+y);
    float3 moved=rounded3(next-high);low=rounded3(y-moved);high=next;
}
void compensated4(inout float4 high,inout float4 low,float4 increment){
    float4 y=rounded4(increment+low);float4 next=rounded4(high+y);
    float4 moved=rounded4(next-high);low=rounded4(y-moved);high=next;
}
float expm1Stable(float x){
    if(abs(x)<.01)return x*(1+x*(.5+x*(1./6+x*(1./24+x/120))));
    return exp(x)-1;
}
float3 expm1Stable3(float3 x){return float3(expm1Stable(x.x),expm1Stable(x.y),expm1Stable(x.z));}
float4 rotationIncrement(float3 w,float dt,float4 q){
    float half=.5*length(w)*dt;
    float coefficient=abs(half)<.001?.5*dt*(1-half*half/6):sin(half)/length(w);
    float cm1=-2*sin(.5*half)*sin(.5*half);
    return qmul(float4(coefficient*w,cm1),q);
}
void accumulate(uint id,float3 increment){
    float3 high=r.accumulated[id].xyz,low=r.accumulatedRemainder[id].xyz;
    compensated3(high,low,increment);
    r.accumulated[id]=float4(high,0);r.accumulatedRemainder[id]=float4(low,0);
}
float3 centerBranch(uint i,uint j){
    float3 box=r.cell[0].xyz;
    float3 d=(r.position[j].xyz-r.position[i].xyz)+(r.positionRemainder[j].xyz-r.positionRemainder[i].xyz);
    return r.c.periodic!=0?d-floor(d/box+.5)*box:d;
}
float3 omega(uint i){
    float3 b=qinv(r.rotation[i],r.momentum[i].xyz+r.momentumRemainder[i].xyz);
    float3 w=float3(dot(r.invI[3*i].xyz,b),dot(r.invI[3*i+1].xyz,b),dot(r.invI[3*i+2].xyz,b));
    return qrot(r.rotation[i],w);
}
float rotationalEnergy(uint i,float3 l){
    float3 b=qinv(r.rotation[i],l);
    float3 w=float3(dot(r.invI[3*i].xyz,b),dot(r.invI[3*i+1].xyz,b),dot(r.invI[3*i+2].xyz,b));
    return .5*dot(b,w);
}
float cellPower(uint i){
    float3 diagonal=float3(r.particleStress[3*i].x,r.particleStress[3*i+1].y,r.particleStress[3*i+2].z);
    float3 v=r.velocity[i].xyz;
    return -dot(diagonal+r.body[i].y*v*v,r.cell[1].xyz);
}
float3 image(float3 d){float3 box=r.cell[0].xyz;return r.c.periodic!=0?d-floor(d/box+0.5)*box:d;}
float3 wrap(float3 x){float3 box=r.cell[0].xyz;return r.c.periodic!=0?x-floor(x/box)*box:x;}
void fail(uint bit){uint unused;InterlockedOr(r.control[0],bit,unused);}
uint3 binIndex(uint i){return min((uint3)floor(r.position[i].xyz/r.cell[0].xyz*float3(r.c.bins)),r.c.bins-1);}
uint binFlat(uint3 b){return (b.x*r.c.bins.y+b.y)*r.c.bins.z+b.z;}
uint modIndex(int a,uint n){return uint((a%int(n)+int(n))%int(n));}

float4 sdf(uint s,float3 p){
    uint4 meta=r.grids[s]; float4 o=r.gridOrigin[s];
    float3 g=(p-o.xyz)/o.w; int3 a=int3(floor(g));
    if(any(a<0)||any(a>=int3(meta.yzw)-1))return float4(0,0,0,1e20);
    float3 t=g-float3(a);float value=0;float3 grad=0;
    for(uint i=0;i<2;i++)for(uint j=0;j<2;j++)for(uint k=0;k<2;k++){
        uint3 b=uint3(i,j,k);float3 w=lerp(1-t,t,float3(b));uint3 index=uint3(a)+b;
        float f=r.distance[meta.x+(index.x*meta.z+index.y)*meta.w+index.z];
        value+=f*w.x*w.y*w.z;
        grad+=f*float3((2*float(i)-1)*w.y*w.z,(2*float(j)-1)*w.x*w.z,(2*float(k)-1)*w.x*w.y)/o.w;
    }
    return float4(grad,value);
}

// Nearest triangle feature: face=0, AB/BC/CA=1/2/3, A/B/C=4/5/6.
float3 closestTriangle(float3 p,float3 a,float3 b,float3 c,out uint feature){
    float3 ab=b-a,ac=c-a,ap=p-a;
    float d1=dot(ab,ap),d2=dot(ac,ap);feature=4;
    if(d1<=0&&d2<=0)return a;
    float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0&&d4<=d3){feature=5;return b;}
    float vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0){feature=1;return a+(d1/(d1-d3))*ab;}
    float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0&&d5<=d6){feature=6;return c;}
    float vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0){feature=3;return a+(d2/(d2-d6))*ac;}
    float va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){feature=2;return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);}
    float inv=1/(va+vb+vc);feature=0;
    return a+ab*(vb*inv)+ac*(vc*inv);
}
float meshBoxDistance(uint node,float3 p){
    float3 d=max(max(r.meshMin[node].xyz-p,p-r.meshMax[node].xyz),0);
    return dot(d,d);
}
float4 meshDistance(uint shape,float3 p){
    uint root=r.meshRoots[shape].x;
    uint pending[32];uint top=1;pending[0]=root;
    float best=1e30;float3 nearest=0,pseudo=0;
    while(top>0){
        uint node=pending[--top];
        if(meshBoxDistance(node,p)>best)continue;
        uint4 link=r.meshLinks[node];
        if(link.z==0){
            uint a=node+1,b=r.meshLinks[a].x;
            float da=meshBoxDistance(a,p),db=meshBoxDistance(b,p);
            if(da<db){uint swap=a;a=b;b=swap;float ds=da;da=db;db=ds;}
            // Far child first; LIFO visits the nearer child next.
            if(top+2>32){fail(1024);return float4(0,0,0,1e20);}
            if(da<=best)pending[top++]=a;
            if(db<=best)pending[top++]=b;
            continue;
        }
        for(uint j=0;j<link.z;j++){
            uint triangle=link.y+j,feature;
            float3 q=closestTriangle(p,r.meshTriangles[3*triangle].xyz,r.meshTriangles[3*triangle+1].xyz,r.meshTriangles[3*triangle+2].xyz,feature);
            float distance=dot(p-q,p-q);
            if(distance<best){best=distance;nearest=q;pseudo=r.meshNormals[7*triangle+feature].xyz;}
        }
    }
    float distance=sqrt(best);float side=dot(p-nearest,pseudo)<0?-1.:1.;
    float3 normal=distance>1e-12?side*(p-nearest)/distance:pseudo;
    return float4(normal,side*distance);
}
[shader("compute")][numthreads(64,1,1)]
void meshProbe(uint id:SV_DispatchThreadID,uniform uint count,uniform uint shape,
               StructuredBuffer<float4> points,RWStructuredBuffer<float4> result){
    if(id<count)result[id]=meshDistance(shape,points[id].xyz);
}

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
    return true;
}

bool meshClusterMayContact(uint src,uint dst,uint cluster){
    float3 sourceLow=r.meshClusterMin[cluster].xyz,sourceHigh=r.meshClusterMax[cluster].xyz;
    float3 branch=centerBranch(src,dst),low=1e30,high=-1e30;
    for(uint corner=0;corner<8;corner++){
        float3 p=float3((corner&1)!=0?sourceHigh.x:sourceLow.x,
                        (corner&2)!=0?sourceHigh.y:sourceLow.y,
                        (corner&4)!=0?sourceHigh.z:sourceLow.z);
        p=qinv(r.rotation[dst],qrot(r.rotation[src],p*r.body[src].x)-branch)/r.body[dst].x;
        low=min(low,p);high=max(high,p);
    }
    // Bound transform/dot rounding as well as the rounded source coordinates.
    float pad=64.*1.1920928955078125e-7*(1.+max(max(abs(low.x),abs(low.y)),abs(low.z))
        +max(max(abs(high.x),abs(high.y)),abs(high.z))+length(branch)/r.body[dst].x);
    low-=pad;high+=pad;
    uint shape=r.shapeId[dst],root=r.meshRoots[shape].x;
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
    return true;
}
[shader("compute")][numthreads(64,1,1)]
void meshClusterProbe(uint id:SV_DispatchThreadID,uniform uint count,
                      uniform uint src,uniform uint dst,RWStructuredBuffer<uint> result){
    if(id<count){
        uint cluster=r.meshClusterInfo[r.shapeId[src]].x+id;
        result[id]=meshClusterMayContact(src,dst,cluster)?1:0;
    }
}
[shader("compute")][numthreads(64,1,1)]
void meshCullProbe(uint id:SV_DispatchThreadID,uniform uint count,uniform uint shape,
                   StructuredBuffer<float4> points,RWStructuredBuffer<uint> result){
    if(id<count)result[id]=meshMayContain(shape,points[id].xyz)?1:0;
}
bool nodeContact(uint src,uint dst,uint node,out float4 pd,out float4 na){
    uint sid=r.shapeId[src];float4 sp=r.sample[r.sampleInfo[sid].x+node];
    float3 arm=qrot(r.rotation[src],sp.xyz*r.body[src].x);
    float3 branch=centerBranch(src,dst);
    float3 local=qinv(r.rotation[dst],arm-branch)/r.body[dst].x;
    uint targetShape=r.shapeId[dst];float4 f;
    if(r.c.meshGeometry!=0){
        if(!meshMayContain(targetShape,local)){pd=0;na=0;return false;}
        f=meshDistance(targetShape,local);
    }else f=sdf(targetShape,local);
    pd=float4(arm,-f.w*r.body[dst].x);na=0;
    if(pd.w<=0)return false;
    float3 grad=qrot(r.rotation[dst],f.xyz);float len=length(grad);
    if(len<1e-7||!isfinite(pd.w)){fail(2);return false;}
    na=float4(grad/len,sp.w*r.body[src].x*r.body[src].x);
    if(pd.w>r.c.maxOverlap*min(r.body[src].w,r.body[dst].w))fail(4);
    return true;
}

[shader("compute")][numthreads(64,1,1)]
void clearBins(uint id:SV_DispatchThreadID){
    if(r.control[4]==0)return;
    if(id<r.c.binCount)r.binCounts[id]=0;
}
[shader("compute")][numthreads(64,1,1)]
void fillBins(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[4]==0)return;
    uint bin=binFlat(binIndex(id));uint slot;InterlockedAdd(r.binCounts[bin],1,slot);
    if(slot<r.c.binCapacity)r.binParticles[bin*r.c.binCapacity+slot]=id;
    else fail(8);
}
uint findPairs(uint id,bool emit){
    uint3 b=binIndex(id);float cutoff=2*r.body[id].z+r.c.skin;
    // Global conservative radius is supplied through referenceCell[1].w.
    cutoff=r.referenceCell[1].w;
    int3 reach=int3(ceil(cutoff/r.cell[0].xyz*float3(r.c.bins)));
    int3 low=max(int3(b)-reach,int3(0));
    int3 high=min(int3(b)+reach,int3(r.c.bins)-1);
    uint3 width=r.c.periodic!=0?min(uint3(2*reach+1),r.c.bins):uint3(high-low+1);uint count=0;
    for(uint x=0;x<width.x;x++)for(uint y=0;y<width.y;y++)for(uint z=0;z<width.z;z++){
        uint3 c=uint3(modIndex(int(b.x)-reach.x+int(x),r.c.bins.x),modIndex(int(b.y)-reach.y+int(y),r.c.bins.y),modIndex(int(b.z)-reach.z+int(z),r.c.bins.z));
        if(r.c.periodic==0)c=uint3(low+int3(x,y,z));
        uint bi=binFlat(c);uint occupancy=min(r.binCounts[bi],r.c.binCapacity);
        for(uint slot=0;slot<occupancy;slot++){
            uint j=r.binParticles[bi*r.c.binCapacity+slot];
            if(j<=id)continue;
            float3 d=centerBranch(id,j);
            float range=r.body[id].z+r.body[j].z+r.c.skin;
            if(dot(d,d)>range*range)continue;
            if(emit){uint index=r.pairOffsets[id]+count;if(index<r.c.pairCapacity)r.pairs[index]=uint2(id,j);else fail(16);}
            count++;
        }
    }
    return count;
}
[shader("compute")][numthreads(64,1,1)]
void countPairs(uint id:SV_DispatchThreadID){if(id<r.c.N&&r.control[4]!=0)r.pairCounts[id]=findPairs(id,false);}
[shader("compute")][numthreads(64,1,1)]
void emitPairs(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[4]==0)return;
    findPairs(id,true);r.referencePosition[id]=r.position[id];r.referencePositionRemainder[id]=r.positionRemainder[id];
    if(id==0){r.control[1]=r.pairOffsets[r.c.N-1]+r.pairCounts[r.c.N-1];r.referenceCell[0]=r.cell[0];}
}

// Exact mesh queries are expensive. One-pass compaction uses integer atomics
// only; persistent friction identity remains (source,target,source_sample).
[shader("compute")][numthreads(64,1,1)]
void resetMeshContacts(uint id:SV_DispatchThreadID){
    if(r.control[0]!=0)return;
    if(id<r.c.pairCapacity)r.contactCounts[id]=0;
    if(id==0){r.control[2]=0;r.control[6]=0;}
}
groupshared uint meshClusterCandidate;
groupshared uint meshGroupActive;
[shader("compute")][numthreads(64,1,1)]
void collectMeshContacts(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint pair=group.x;
    if(pair>=min(r.control[1],r.c.pairCapacity))return;
    if(lane==0)meshGroupActive=r.control[0]==0?1:0;
    GroupMemoryBarrierWithGroupSync();
    if(meshGroupActive==0)return;
    // Every lane reaches every barrier, including when another group flags an
    // overflow. No lane may return early from a contact-capacity failure.
    uint2 ij=r.pairs[pair];
    for(uint pass=0;pass<2;pass++){
        uint i=pass==0?ij.x:ij.y,j=pass==0?ij.y:ij.x;
        uint4 info=r.meshClusterInfo[r.shapeId[i]];
        uint cluster=info.x;
        while(cluster<info.x+info.y){
            if(lane==0)meshClusterCandidate=meshClusterMayContact(i,j,cluster)?1:0;
            GroupMemoryBarrierWithGroupSync();
            uint candidate=meshClusterCandidate;
            uint4 range=r.meshClusterRanges[cluster];
            uint nextCluster=candidate!=0?cluster+1:range.z;
            if(candidate!=0&&lane<range.y){
                uint node=r.meshSampleIndices[range.x+lane];float4 pd,na;
                if(nodeContact(i,j,node,pd,na)){
                    uint offset;InterlockedAdd(r.control[2],1,offset);
                    if(offset>=r.c.contactCapacity)fail(32);
                    else{
                        r.meta[offset]=uint4(i,j,node,0);r.pointDepth[offset]=pd;r.normalArea[offset]=na;
                        uint unused;InterlockedAdd(r.contactCounts[pair],1,unused);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            // Uniform, stackless preorder traversal. All lanes make the same
            // decision and reach every barrier, also for rejected subtrees.
            cluster=nextCluster;
        }
    }
}

groupshared uint localCount[64];
[shader("compute")][numthreads(64,1,1)]
void countContacts(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint pair=group.x;
    if(pair>=r.c.pairCapacity)return;
    uint count=0;
    if(pair<r.control[1]&&r.control[0]==0){
        uint2 ij=r.pairs[pair];uint a=r.sampleInfo[r.shapeId[ij.x]].y;uint total=a+r.sampleInfo[r.shapeId[ij.y]].y;
        for(uint s=lane;s<total;s+=64){uint i=s<a?ij.x:ij.y;uint j=s<a?ij.y:ij.x;uint node=s<a?s:s-a;float4 pd,na;if(nodeContact(i,j,node,pd,na))count++;}
    }
    localCount[lane]=count;GroupMemoryBarrierWithGroupSync();
    for(uint shift=32;shift>0;shift/=2){if(lane<shift)localCount[lane]+=localCount[lane+shift];GroupMemoryBarrierWithGroupSync();}
    if(lane==0)r.contactCounts[pair]=localCount[0];
}
[shader("compute")][numthreads(64,1,1)]
void emitContacts(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint pair=group.x;
    if(pair>=min(r.control[1],r.c.pairCapacity)||r.control[0]!=0)return;
    uint2 ij=r.pairs[pair];uint a=r.sampleInfo[r.shapeId[ij.x]].y;uint total=a+r.sampleInfo[r.shapeId[ij.y]].y;uint count=0;
    for(uint s=lane;s<total;s+=64){float4 pd,na;if(nodeContact(s<a?ij.x:ij.y,s<a?ij.y:ij.x,s<a?s:s-a,pd,na))count++;}
    localCount[lane]=count;GroupMemoryBarrierWithGroupSync();
    for(uint shift=1;shift<64;shift*=2){uint v=lane>=shift?localCount[lane-shift]:0;GroupMemoryBarrierWithGroupSync();localCount[lane]+=v;GroupMemoryBarrierWithGroupSync();}
    uint offset=r.contactOffsets[pair]+localCount[lane]-count;
    for(uint s=lane;s<total;s+=64){uint i=s<a?ij.x:ij.y;uint j=s<a?ij.y:ij.x;uint node=s<a?s:s-a;float4 pd,na;
        if(nodeContact(i,j,node,pd,na)){
            if(offset>=r.c.contactCapacity){fail(32);return;}
            r.meta[offset]=uint4(i,j,node,0);r.pointDepth[offset]=pd;r.normalArea[offset]=na;offset++;
        }
    }
}
[shader("compute")][numthreads(1,1,1)]
void contactTotal(uint id:SV_DispatchThreadID){
    uint n=r.c.pairCapacity;
    r.control[2]=r.contactOffsets[n-1]+r.contactCounts[n-1];
    r.control[6]=0;
    if(r.control[2]>r.c.contactCapacity)fail(32);
}

uint hashKey(uint4 key){uint h=key.x*73856093u^key.y*19349663u^key.z*83492791u;h^=h>>16;return h&(r.c.hashSize-1);}
bool keyEqual(uint4 a,uint4 b){return all(a.xyz==b.xyz);}
float3 transported(float3 x,float3 a,float3 b,float spin){
    float c=clamp(dot(a,b),-1.,1.);if(c<-.999999)return 0;
    float3 v=cross(a,b);float3 out=x+cross(v,x)+cross(v,cross(v,x))/(1+c);
    out=out*cos(spin)+cross(b,out)*sin(spin)+b*dot(b,out)*(1-cos(spin));
    return out-b*dot(out,b);
}
[shader("compute")][numthreads(64,1,1)]
void clearContactStorage(uint id:SV_DispatchThreadID){
    if(id<r.c.hashSize)r.hash[id]=EMPTY;
    if(id<r.c.N)r.heads[id]=EMPTY;
}
[shader("compute")][numthreads(64,1,1)]
void resolveContacts(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint4 key=r.meta[id];uint i=key.x,j=key.y;float3 n=r.normalArea[id].xyz;
    float3 wi=omega(i),wj=omega(j);float3 elastic=0;uint slot=hashKey(key);
    if(r.c.historyStep!=0){
        for(uint probe=0;probe<r.c.hashSize;probe++){
            uint old=r.oldHash[slot];if(old==EMPTY)break;
            if(keyEqual(key,r.oldMeta[old])){elastic=transported(r.oldSlip[old].xyz,r.oldNormal[old].xyz,n,.5*dot(wi+wj,n)*r.c.dt);break;}
            slot=(slot+1)&(r.c.hashSize-1);
        }
    }
    float3 arm=r.pointDepth[id].xyz;float3 branch=centerBranch(i,j);
    float3 rv=r.velocity[i].xyz-r.velocity[j].xyz-r.cell[1].xyz*branch+cross(wi,arm)-cross(wj,arm-branch);
    elastic+=(rv-n*dot(rv,n))*r.c.dt;
    float3 trial=elastic;
    float area=r.normalArea[id].w,depth=r.pointDepth[id].w;
    float fn=.5*r.c.kn*area*depth;float3 ft=-.5*r.c.kt*area*elastic;float magnitude=length(ft);
    if(magnitude>r.c.mu*fn&&magnitude>0){ft*=r.c.mu*fn/magnitude;elastic=-ft/(.5*r.c.kt*area);uint unused;InterlockedAdd(r.control[6],1,unused);}
    r.slip[id]=float4(elastic,max(0.,-dot(ft,trial-elastic)));
    r.contactForce[id]=float4(fn*n+ft,.25*area*(r.c.kn*depth*depth+r.c.kt*dot(elastic,elastic)));
    uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);r.next[2*id]=oldHead;
    InterlockedExchange(r.heads[j],2*id+1,oldHead);r.next[2*id+1]=oldHead;
}
[shader("compute")][numthreads(64,1,1)]
void buildHistoryHash(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint slot=hashKey(r.meta[id]);
    for(uint probe=0;probe<r.c.hashSize;probe++){
        uint old;InterlockedCompareExchange(r.hash[slot],EMPTY,id,old);if(old==EMPTY)return;
        slot=(slot+1)&(r.c.hashSize-1);
    }
    fail(64);
}
[shader("compute")][numthreads(64,1,1)]
void gather(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0)return;
    float3 f=0,t=0;float3 stress0=0,stress1=0,stress2=0;float energy=0,denom=0,maxDepth=0,dissipation=0;
    uint entry=r.heads[id];uint visited=0;
    while(entry!=EMPTY){
        if(visited++>2*r.c.contactCapacity){fail(128);return;}
        uint contact=entry/2;bool source=(entry&1)==0;uint4 key=r.meta[contact];
        float3 force=r.contactForce[contact].xyz;float3 branch=centerBranch(key.x,key.y);
        float3 arm=r.pointDepth[contact].xyz;
        f+=source?force:-force;t+=source?cross(arm,force):cross(arm-branch,-force);
        // Compression-positive virial: -force_on_source outer centre branch.
        stress0-=.5*force.x*branch;stress1-=.5*force.y*branch;stress2-=.5*force.z*branch;
        energy+=.5*r.contactForce[contact].w;denom+=length(force);
        maxDepth=max(maxDepth,r.pointDepth[contact].w/min(r.body[key.x].w,r.body[key.y].w));
        dissipation+=.5*r.slip[contact].w;
        entry=r.next[entry];
    }
    r.force[id]=float4(f,maxDepth);r.torque[id]=float4(t,dissipation);
    r.particleStress[3*id]=float4(stress0,0);r.particleStress[3*id+1]=float4(stress1,0);r.particleStress[3*id+2]=float4(stress2,0);
    float kinetic=.5*(r.body[id].y*dot(r.velocity[id].xyz,r.velocity[id].xyz)+dot(r.momentum[id].xyz,omega(id)));
    r.particleDiagnostic[id]=float4(kinetic,energy,length(f),denom);
    accumulate(id,float3(dissipation,0,0));
}
[shader("compute")][numthreads(64,1,1)]
void beginStep(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0)return;
    float dt=r.c.dt;float3 rate=r.cell[1].xyz;
    float3 v=r.velocity[id].xyz,vl=r.velocityRemainder[id].xyz;
    float3 l=r.momentum[id].xyz,ll=r.momentumRemainder[id].xyz;
    compensated3(v,vl,.5*dt*r.force[id].xyz/r.body[id].y);
    compensated3(l,ll,.5*dt*r.torque[id].xyz);
    float loss=-expm1Stable(-r.c.drag*dt)*(.5*r.body[id].y*dot(v+vl,v+vl)+rotationalEnergy(id,l+ll));
    accumulate(id,float3(0,loss,.5*dt*cellPower(id)));
    float damping=expm1Stable(-.5*r.c.drag*dt);
    compensated3(v,vl,damping*(v+vl));compensated3(l,ll,damping*(l+ll));
    r.momentum[id]=float4(l,0);r.momentumRemainder[id]=float4(ll,0);
    float4 q=r.rotation[id],ql=r.rotationRemainder[id];float3 w=omega(id);
    r.rotation[id]=normalize(q+rotationIncrement(w,.5*dt,q));w=omega(id);
    compensated4(q,ql,rotationIncrement(w,dt,q));
    if(abs(dot(q,q)-1)>2e-6){q=normalize(q);ql=0;}
    r.rotation[id]=q;r.rotationRemainder[id]=ql;
    float3 logarithm=r.cellLog[0].xyz,residual=r.cellLog[1].xyz;
    compensated3(logarithm,residual,rate*dt);
    float3 box=r.cell[2].xyz*exp(logarithm);
    float3 p=r.position[id].xyz,pl=r.positionRemainder[id].xyz;
    compensated3(p,pl,expm1Stable3(rate*dt)*(p+pl)+dt*(v+vl));
    if(any(!isfinite(p))||any(!isfinite(pl))){fail(512);return;}
    if(r.c.periodic!=0)p-=floor(p/box)*box;
    else if(any(p+pl<0)||any((p-box)+pl>=0)){fail(2048);return;}
    r.position[id]=float4(p,0);r.positionRemainder[id]=float4(pl,0);
    compensated3(v,vl,expm1Stable3(-rate*dt)*(v+vl));
    r.velocity[id]=float4(v,0);r.velocityRemainder[id]=float4(vl,0);
}
[shader("compute")][numthreads(1,1,1)]
void advanceCell(uint id:SV_DispatchThreadID){
    if(r.control[0]!=0)return;
    float3 logarithm=r.cellLog[0].xyz,residual=r.cellLog[1].xyz;
    compensated3(logarithm,residual,r.cell[1].xyz*r.c.dt);
    r.cellLog[0]=float4(logarithm,0);r.cellLog[1]=float4(residual,0);
    r.cell[0]=float4(r.cell[2].xyz*exp(logarithm),0);r.control[4]=0;
    if(r.c.periodic!=0&&r.referenceCell[1].w>=.5*min(r.cell[0].x,min(r.cell[0].y,r.cell[0].z)))fail(256);
}
[shader("compute")][numthreads(64,1,1)]
void checkRebuild(uint id:SV_DispatchThreadID){
    if(id>=r.c.N)return;
    float3 d=image((r.position[id].xyz-r.referencePosition[id].xyz)+(r.positionRemainder[id].xyz-r.referencePositionRemainder[id].xyz));
    float deformation=length(r.cell[0].xyz-r.referenceCell[0].xyz);
    if(length(d)+deformation>.5*r.c.skin){uint unused;InterlockedOr(r.control[4],1,unused);}
}
[shader("compute")][numthreads(64,1,1)]
void finishStep(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0)return;
    float3 v=r.velocity[id].xyz,vl=r.velocityRemainder[id].xyz;
    float3 l=r.momentum[id].xyz,ll=r.momentumRemainder[id].xyz;
    compensated3(v,vl,.5*r.c.dt*r.force[id].xyz/r.body[id].y);
    compensated3(l,ll,.5*r.c.dt*r.torque[id].xyz);
    float loss=-expm1Stable(-r.c.drag*r.c.dt)*(.5*r.body[id].y*dot(v+vl,v+vl)+rotationalEnergy(id,l+ll));
    accumulate(id,float3(0,loss,.5*r.c.dt*cellPower(id)));
    float damping=expm1Stable(-.5*r.c.drag*r.c.dt);
    compensated3(v,vl,damping*(v+vl));compensated3(l,ll,damping*(l+ll));
    if(any(!isfinite(v))||any(!isfinite(l))){fail(512);return;}
    r.velocity[id]=float4(v,0);r.velocityRemainder[id]=float4(vl,0);
    r.momentum[id]=float4(l,0);r.momentumRemainder[id]=float4(ll,0);
}

[shader("compute")][numthreads(64,1,1)]
void diagnosticRows(uint id:SV_DispatchThreadID){
    if(id>=r.c.N)return;
    float3 v=r.velocity[id].xyz;float m=r.body[id].y;
    float kinetic=.5*(m*dot(v,v)+dot(r.momentum[id].xyz,omega(id)));
    r.diagnosticRows[id*7]=float4(r.particleStress[id*3].xyz,kinetic);
    r.diagnosticRows[id*7+1]=float4(r.particleStress[id*3+1].xyz,r.particleDiagnostic[id].y);
    r.diagnosticRows[id*7+2]=float4(r.particleStress[id*3+2].xyz,length(r.force[id].xyz));
    r.diagnosticRows[id*7+3]=float4(m*v.x*v,r.particleDiagnostic[id].w);
    r.diagnosticRows[id*7+4]=float4(m*v.y*v,r.force[id].w);
    r.diagnosticRows[id*7+5]=float4(m*v.z*v,r.torque[id].w);
    r.diagnosticRows[id*7+6]=r.accumulated[id];
}

[shader("compute")][numthreads(64,1,1)]
void fabricRows(uint id:SV_DispatchThreadID){
    if(id>=r.c.pairCapacity)return;
    float4 a=0,b=0;
    if(id<r.control[1]&&r.contactCounts[id]>0){
        uint2 ij=r.pairs[id];float3 n=normalize(centerBranch(ij.x,ij.y));
        a=float4(n*n,n.x*n.y);b=float4(n.x*n.z,n.y*n.z,1,0);
    }
    r.fabricRows[2*id]=a;r.fabricRows[2*id+1]=b;
}
