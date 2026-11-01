#include "plane.slang"

// Standalone Slang kernels. Quaternion order xyzw; body-to-world rotations.
[shader("compute")]
[numthreads(64,1,1)]
void smoke(uint tid: SV_DispatchThreadID, uniform uint N,
           StructuredBuffer<float> a, RWStructuredBuffer<float> b)
{
    if (tid < N) b[tid] = a[tid]*a[tid]+1.0;
}

struct Config {
    uint activeTimestep; uint N; uint shapes; uint pairCapacity; uint contactCapacity;
    uint binCapacity; uint binCount; uint3 bins; uint hashSize;
    float dt; float kn; float kt; float mu; float normalViscosity;
    float skin; float drag; float maxOverlap; uint historyStep;
    uint meshGeometry; uint periodic; uint wallCount; uint wallBoxEnabled;
};
struct Resources {
    Config c;
    RWStructuredBuffer<uint4> adjCacheKeys;
    RWStructuredBuffer<uint> adjCacheCounts,adjCacheMetrics;
    RWStructuredBuffer<uint> adjCounts,adjOffsets,adjCursor,adjEntries;
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
    StructuredBuffer<uint4> meshFrontier;
    RWStructuredBuffer<uint> meshPairMask;
    RWStructuredBuffer<float4> meshPairTransform;
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
    RWStructuredBuffer<float> normalLossPower;
    RWStructuredBuffer<float4> normalDamping; // cumulative high,low; current power
    RWStructuredBuffer<float4> forceVelocity; // committed contact-evaluation kinematics
    RWStructuredBuffer<float4> forceOmega;
    StructuredBuffer<uint4> oldMeta;
    StructuredBuffer<float4> oldSlip;
    StructuredBuffer<float4> oldNormal;
    StructuredBuffer<uint> oldHash;
    RWStructuredBuffer<uint> hash;
    RWStructuredBuffer<uint> heads;
    RWStructuredBuffer<uint> next;
    RWStructuredBuffer<float4> particleStress; // 3 rows per particle, contact contribution
    RWStructuredBuffer<float4> timestepRows; // frozen-contact elastic frequency^2, viscous rate
    RWStructuredBuffer<float4> particleDiagnostic; // kinetic, elastic proxy, force norm, contact force norms
    RWStructuredBuffer<float4> diagnosticRows;
    RWStructuredBuffer<float4> accumulated; // friction loss, drag loss, cell work, reserved
    RWStructuredBuffer<float4> fabricRows;
    RWStructuredBuffer<float4> wallPlane; // unit inward normal, normalized offset
    RWStructuredBuffer<float4> wallRemainder; // compensated offset in w
    RWStructuredBuffer<float4> wallVelocity;
    StructuredBuffer<float4> wallMaterial; // kn,kt,mu
    StructuredBuffer<uint> wallBoxIds;
    RWStructuredBuffer<float4> wallRows; // wall-major, body-major, three rows
    RWStructuredBuffer<float4> wallAccumulated; // compensated wall work high,low

};
ParameterBlock<Resources> r;
bool denseGather(){return r.control[2]>=64*r.c.N;}
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
void accumulate(uint id,float4 increment){
    float4 high=r.accumulated[id],low=r.accumulatedRemainder[id];
    compensated4(high,low,increment);
    r.accumulated[id]=high;r.accumulatedRemainder[id]=low;
}
float wallStepWork(uint id,float dt){
    float total=0;
    for(uint w=0;w<r.c.wallCount;w++){
        uint index=w*r.c.N+id;
        float increment=dt*r.wallRows[3*index+1].w;
        float4 state=r.wallAccumulated[index];
        float3 high=float3(state.x,0,0),low=float3(state.y,0,0);
        compensated3(high,low,float3(increment,0,0));
        r.wallAccumulated[index]=float4(high.x,low.x,0,0);
        r.wallRows[3*index+2]=float4(high.x+low.x,0,0,0);
        total+=increment;
    }
    return total;
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
void accumulateNormalLoss(uint id,float dt){
    float4 value=r.normalDamping[id];
    float3 high=float3(value.x,0,0),low=float3(value.y,0,0);
    compensated3(high,low,float3(value.z*dt,0,0));
    value.x=high.x;value.y=low.x;r.normalDamping[id]=value;
}
[shader("compute")][numthreads(64,1,1)]
void captureContactKinematics(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0)return;
    r.forceVelocity[id]=float4(r.velocity[id].xyz+r.velocityRemainder[id].xyz,0);
    r.forceOmega[id]=float4(omega(id),0);
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
    return meshBoxMayContact(r.shapeId[dst],low,high);
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
    float3 p=r.position[id].xyz,pl=r.positionRemainder[id].xyz;
    if(any(!isfinite(p))||any(!isfinite(pl))){fail(512);return;}
    if(r.c.periodic==0&&(any(p+pl<0)||any((p-r.cell[0].xyz)+pl>=0))){fail(2048);return;}
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
    if(id==0){
        r.control[2]=0;r.control[6]=0;
    }
}
void meshPairBounds(uint direction,uint cluster,out float3 low,out float3 high){
    float3 center=.5*(r.meshClusterMin[cluster].xyz+r.meshClusterMax[cluster].xyz);
    float3 half=.5*(r.meshClusterMax[cluster].xyz-r.meshClusterMin[cluster].xyz);
    float3 c,e;
    for(uint axis=0;axis<3;axis++){
        float4 row=r.meshPairTransform[4*direction+axis];
        c[axis]=dot(row.xyz,center)+row.w;e[axis]=dot(abs(row.xyz),half);
    }
    low=c-e;high=c+e;
    // Cover composed-matrix, center/extent and dot rounding conservatively.
    float pad=256.*1.1920928955078125e-7*(1.+max(max(abs(low.x),abs(low.y)),abs(low.z))
        +max(max(abs(high.x),abs(high.y)),abs(high.z))+r.meshPairTransform[4*direction+3].x);
    low-=pad;high+=pad;
}
[shader("compute")][numthreads(64,1,1)]
void meshPairBoundsProbe(uint id:SV_DispatchThreadID,uniform uint count,uniform uint direction,
                         uniform uint first,RWStructuredBuffer<float4> result){
    if(id>=count)return;
    float3 low,high;meshPairBounds(direction,first+id,low,high);
    result[2*id]=float4(low,0);result[2*id+1]=float4(high,0);
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

// Root rejection is shared by all subtree workgroups for each pair.
[shader("compute")][numthreads(64,1,1)]
void prepareMeshPairs(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[1],r.c.pairCapacity)||r.control[0]!=0)return;
    uint2 ij=r.pairs[id];uint mask=0;
    if(meshClusterMayContact(ij.x,ij.y,r.meshClusterInfo[r.shapeId[ij.x]].x))mask|=1;
    if(meshClusterMayContact(ij.y,ij.x,r.meshClusterInfo[r.shapeId[ij.y]].x))mask|=2;
    for(uint pass=0;pass<2;pass++){
        if((mask&(1u<<pass))==0)continue;
        uint i=pass==0?ij.x:ij.y,j=pass==0?ij.y:ij.x;
        float ratio=r.body[i].x/r.body[j].x;
        float3 a=qinv(r.rotation[j],qrot(r.rotation[i],float3(ratio,0,0)));
        float3 b=qinv(r.rotation[j],qrot(r.rotation[i],float3(0,ratio,0)));
        float3 c=qinv(r.rotation[j],qrot(r.rotation[i],float3(0,0,ratio)));
        float3 branch=centerBranch(i,j),offset=qinv(r.rotation[j],-branch)/r.body[j].x;
        uint base=4*(2*id+pass);
        r.meshPairTransform[base]=float4(a.x,b.x,c.x,offset.x);
        r.meshPairTransform[base+1]=float4(a.y,b.y,c.y,offset.y);
        r.meshPairTransform[base+2]=float4(a.z,b.z,c.z,offset.z);
        r.meshPairTransform[base+3]=float4(length(branch)/r.body[j].x,0,0,0);
    }
    r.meshPairMask[id]=mask;
}

void collectMeshSubtree(uint pair,uint pass,uint tile,uint lane){
    if(pair>=min(r.control[1],r.c.pairCapacity))return;
    if((r.meshPairMask[pair]&(1u<<pass))==0)return;
    uint2 ij=r.pairs[pair];
    uint i=pass==0?ij.x:ij.y,j=pass==0?ij.y:ij.x;
    uint4 subtree=r.meshFrontier[32*r.shapeId[i]+tile];
    if(subtree.y==0)return;
    if(lane==0)meshGroupActive=r.control[0]==0?1:0;
    GroupMemoryBarrierWithGroupSync();
    if(meshGroupActive==0)return;
    uint cluster=subtree.x;
    while(cluster<subtree.y){
        if(lane==0){
            float3 low,high;meshPairBounds(2*pair+pass,cluster,low,high);
            meshClusterCandidate=meshBoxMayContact(r.shapeId[j],low,high)?1:0;
        }
        GroupMemoryBarrierWithGroupSync();
        uint candidate=meshClusterCandidate;
        uint4 range=r.meshClusterRanges[cluster];
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
        cluster=candidate!=0?cluster+1:range.z;
    }
}

// Metal currently lacks indirect dispatch in this backend. A bounded persistent
// pool consumes all GPU-known tasks, independent of reserved pair capacity.
[shader("compute")][numthreads(64,1,1)]
void collectMeshFrontiers(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint tasks=64*min(r.control[1],r.c.pairCapacity);
    for(uint task=group.x;task<tasks;task+=4096){
        collectMeshSubtree(task/64,(task%64)/32,task%32,lane);
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


// Two-component sums retain cancellation residuals under Metal fast math.
// Materialization barriers keep these error terms from being reassociated away.
float rounded1(float x){return asfloat(asuint(x)^r.control[15]);}
float2 wallTwoSum(float a,float b){
    float high=rounded1(a+b),v=rounded1(high-a);
    float low=rounded1(rounded1(a-rounded1(high-v))+rounded1(b-v));
    return float2(high,low);
}
float2 wallAdd(float2 a,float b){
    float2 sum=wallTwoSum(a.x,b);
    return wallTwoSum(sum.x,rounded1(sum.y+a.y));
}
float2 wallAddDot(float2 sum,float3 a,float3 b){
    float3 product=rounded3(a*b);
    float3 error=rounded3(fma(a,b,-product));
    for(uint axis=0;axis<3;axis++){
        sum=wallAdd(sum,product[axis]);sum=wallAdd(sum,error[axis]);
    }
    return sum;
}
// Local plane offset at the grain centre; never form a world-space surface point.
float2 wallCentreDistance(uint body,uint wall){
    float4 plane=r.wallPlane[wall],low=r.wallRemainder[wall];
    float3 p=r.position[body].xyz,pl=r.positionRemainder[body].xyz;
    float2 sum=float2(-plane.w,-low.w);
    sum=wallAddDot(sum,plane.xyz,p);
    sum=wallAddDot(sum,plane.xyz,pl);
    sum=wallAddDot(sum,low.xyz,p);
    return wallAddDot(sum,low.xyz,pl);
}
float localWallDistance(uint body,uint wall,float3 arm){
    float2 sum=wallCentreDistance(body,wall);
    sum=wallAddDot(sum,r.wallPlane[wall].xyz,arm);
    sum=wallAddDot(sum,r.wallRemainder[wall].xyz,arm);
    return rounded1(sum.x+sum.y);
}

// Conservative fixed-normal plane bound over the reusable source-sample tree.
bool wallClusterMayContact(uint body,uint wall,uint cluster){
    float3 low=r.meshClusterMin[cluster].xyz,high=r.meshClusterMax[cluster].xyz;
    float4 plane=r.wallPlane[wall];
    float3 normal=qinv(r.rotation[body],plane.xyz)*r.body[body].x;
    float3 center=.5*(low+high),half=.5*(high-low);
    float2 centreDistance=wallCentreDistance(body,wall);
    float base=centreDistance.x+centreDistance.y;
    float support=dot(normal,center)-dot(abs(normal),half);
    float pad=64.*1.1920928955078125e-7*(1.+abs(plane.w)+length(r.position[body].xyz)
        +length(normal)*(length(center)+length(half)));
    return base+support<=pad;
}
void appendWallContact(uint body,uint wall,uint node){
    float4 point=r.sample[r.sampleInfo[r.shapeId[body]].x+node];
    float3 arm=qrot(r.rotation[body],point.xyz*r.body[body].x);
    float4 plane=r.wallPlane[wall];
    float phi=localWallDistance(body,wall,arm);
    if(!isfinite(phi)){fail(512);return;}
    if(phi>=0)return;
    if(-phi>r.c.maxOverlap*r.body[body].w)fail(4);
    uint offset;InterlockedAdd(r.control[2],1,offset);
    if(offset>=r.c.contactCapacity){fail(32);return;}
    r.meta[offset]=uint4(body,r.c.N+wall,node,1);
    r.pointDepth[offset]=float4(arm,-phi);
    r.normalArea[offset]=float4(plane.xyz,point.w*r.body[body].x*r.body[body].x);
}
groupshared uint wallCandidate;
groupshared uint wallGroupActive;
[shader("compute")][numthreads(64,1,1)]
void collectWallContacts(uint3 group:SV_GroupID,uint lane:SV_GroupIndex){
    uint body=group.x%r.c.N,wall=group.x/r.c.N;
    if(wall>=r.c.wallCount)return;
    if(lane==0)wallGroupActive=r.control[0]==0?1:0;
    GroupMemoryBarrierWithGroupSync();
    if(wallGroupActive==0)return;
    float4 plane=r.wallPlane[wall];
    float2 centreDistance=wallCentreDistance(body,wall);
    float phi=centreDistance.x+centreDistance.y;
    float pad=64.*1.1920928955078125e-7*(1.+length(r.position[body].xyz)+abs(plane.w));
    if(phi>r.body[body].z+pad)return;
    if(r.c.meshGeometry==0){
        uint count=r.sampleInfo[r.shapeId[body]].y;
        for(uint node=lane;node<count;node+=64)appendWallContact(body,wall,node);
        return;
    }
    uint4 info=r.meshClusterInfo[r.shapeId[body]];uint cluster=info.x;
    while(cluster<info.x+info.y){
        if(lane==0)wallCandidate=wallClusterMayContact(body,wall,cluster)?1:0;
        GroupMemoryBarrierWithGroupSync();
        uint candidate=wallCandidate;uint4 range=r.meshClusterRanges[cluster];
        if(candidate!=0&&lane<range.y)appendWallContact(body,wall,r.meshSampleIndices[range.x+lane]);
        GroupMemoryBarrierWithGroupSync();
        cluster=candidate!=0?cluster+1:range.z;
    }
}
[shader("compute")][numthreads(1,1,1)]
void advanceWalls(uint id:SV_DispatchThreadID){
    if(r.control[0]!=0)return;
    for(uint w=0;w<r.c.wallCount;w++){
        float4 plane=r.wallPlane[w],residual=r.wallRemainder[w];
        float3 high=float3(plane.w,0,0),low=float3(residual.w,0,0);
        compensated3(high,low,float3(dot(plane.xyz,r.wallVelocity[w].xyz)*r.c.dt,0,0));
        plane.w=high.x;residual.w=low.x;
        if(!isfinite(high.x)||!isfinite(low.x)){fail(512);return;}
        r.wallPlane[w]=plane;r.wallRemainder[w]=residual;
    }
    if(r.c.wallBoxEnabled!=0){
        for(uint axis=0;axis<3;axis++){
            uint lo=r.wallBoxIds[2*axis],hi=r.wallBoxIds[2*axis+1];
            float low=r.wallPlane[lo].w+r.wallRemainder[lo].w;
            float high=-(r.wallPlane[hi].w+r.wallRemainder[hi].w);
            if(low<0||high>r.cell[0][axis]||high<=low)fail(4096);
        }
    }
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
    if(id<r.c.N){r.heads[id]=EMPTY;r.adjCounts[id]=0;r.adjCursor[id]=0;}
}
[shader("compute")][numthreads(64,1,1)]
void resolveContacts(uint id:SV_DispatchThreadID){
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint4 key=r.meta[id];uint i=key.x,j=key.y;float3 n=r.normalArea[id].xyz;
    bool wall=j>=r.c.N;
    float viscosity=r.c.normalViscosity;if(wall)viscosity=r.wallMaterial[j-r.c.N].w;
    float3 wi=viscosity>0?r.forceOmega[i].xyz:omega(i),wj=0;
    if(!wall)wj=viscosity>0?r.forceOmega[j].xyz:omega(j);
    float3 elastic=0;uint slot=hashKey(key);
    if(r.c.historyStep!=0){
        for(uint probe=0;probe<r.c.hashSize;probe++){
            uint old=r.oldHash[slot];if(old==EMPTY)break;
            if(keyEqual(key,r.oldMeta[old])){elastic=transported(r.oldSlip[old].xyz,r.oldNormal[old].xyz,n,.5*dot(wi+wj,n)*r.c.dt);break;}
            slot=(slot+1)&(r.c.hashSize-1);
        }
    }
    float3 arm=r.pointDepth[id].xyz;
    if(wall){
        uint w=j-r.c.N;float4 material=r.wallMaterial[w];
        float3 v=viscosity>0?r.forceVelocity[i].xyz:r.velocity[i].xyz+r.velocityRemainder[i].xyz;
        float3 rv=v+cross(wi,arm)-r.wallVelocity[w].xyz;
        PlaneResponse response=planeViscousResponse(n,-r.pointDepth[id].w,r.normalArea[id].w,rv,elastic,
            material.x,material.y,material.z,material.w,r.c.dt);
        r.slip[id]=response.slipLoss;r.contactForce[id]=response.forceEnergy;
        r.normalLossPower[id]=response.normalLossPower;
        if(any(!isfinite(response.forceEnergy))||!isfinite(response.normalLossPower)){fail(512);return;}
        if(response.sliding){uint unused;InterlockedAdd(r.control[6],1,unused);}
        uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);
        r.next[2*id]=oldHead;r.next[2*id+1]=EMPTY;uint unused;InterlockedAdd(r.adjCounts[i],1,unused);
        return;
    }
    float3 branch=centerBranch(i,j);
    float3 translation=r.velocity[i].xyz-r.velocity[j].xyz;
    if(viscosity>0)translation=r.forceVelocity[i].xyz-r.forceVelocity[j].xyz;
    float3 rv=translation-r.cell[1].xyz*branch+cross(wi,arm)-cross(wj,arm-branch);
    elastic+=(rv-n*dot(rv,n))*r.c.dt;
    float3 trial=elastic;
    float area=r.normalArea[id].w,depth=r.pointDepth[id].w;
    NormalResponse response=normalResponse(depth,area,dot(rv,n),r.c.kn,viscosity,.5);
    float fn=response.force;r.normalLossPower[id]=response.lossPower;
    float3 ft=-.5*r.c.kt*area*elastic;float magnitude=length(ft);
    if(magnitude>r.c.mu*fn&&magnitude>0){ft*=r.c.mu*fn/magnitude;elastic=-ft/(.5*r.c.kt*area);uint unused;InterlockedAdd(r.control[6],1,unused);}
    r.slip[id]=float4(elastic,max(0.,-dot(ft,trial-elastic)));
    r.contactForce[id]=float4(fn*n+ft,.25*area*(r.c.kn*depth*depth+r.c.kt*dot(elastic,elastic)));
    if(any(!isfinite(r.contactForce[id]))||!isfinite(response.lossPower)){fail(512);return;}
    if(denseGather()){
        uint unused;InterlockedAdd(r.adjCounts[i],1,unused);InterlockedAdd(r.adjCounts[j],1,unused);
    }else{
        uint oldHead;InterlockedExchange(r.heads[i],2*id,oldHead);r.next[2*id]=oldHead;
        InterlockedExchange(r.heads[j],2*id+1,oldHead);r.next[2*id+1]=oldHead;
    }
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
void scatterAdjacency(uint id:SV_DispatchThreadID){
    if(!denseGather())return;
    if(id>=min(r.control[2],r.c.contactCapacity)||r.control[0]!=0)return;
    uint4 key=r.meta[id];uint slot;
    InterlockedAdd(r.adjCursor[key.x],1,slot);r.adjEntries[r.adjOffsets[key.x]+slot]=2*id;
    if(key.y<r.c.N){InterlockedAdd(r.adjCursor[key.y],1,slot);r.adjEntries[r.adjOffsets[key.y]+slot]=2*id+1;}
}
groupshared float4 sharedF[64];
groupshared float4 sharedFL[64];
groupshared float4 sharedT[64];
groupshared float4 sharedTL[64];
groupshared float4 sharedS0[64];
groupshared float4 sharedS1[64];
groupshared float4 sharedS2[64];
groupshared float4 sharedA[64];
void gatherDense(uint id,uint lane){
    if(id>=r.c.N||r.control[0]!=0)return;
    float3 f=0,t=0,fLow=0,tLow=0;float3 stress0=0,stress1=0,stress2=0;float energy=0,denom=0,maxDepth=0,dissipation=0,normalPower=0;
    float stiffnessRate=0,viscousRate=0;
    float inverseInertia=max(max(dot(abs(r.invI[3*id].xyz),float3(1)),dot(abs(r.invI[3*id+1].xyz),float3(1))),dot(abs(r.invI[3*id+2].xyz),float3(1)));
    for(uint index=lane;index<r.adjCounts[id];index+=64){
        uint entry=r.adjEntries[r.adjOffsets[id]+index];
        uint contact=entry/2;bool source=(entry&1)==0;uint4 key=r.meta[contact];
        float3 force=r.contactForce[contact].xyz;
        float3 arm=r.pointDepth[contact].xyz;
        if(r.c.activeTimestep!=0){
            bool wall=key.y>=r.c.N;
            float3 localArm=source?arm:arm-centerBranch(key.x,key.y);
            // Spectral bound for frozen contact geometry in mass-normalized 6D.
            // Coulomb coupling adds mu*kn (and mu*viscosity).
            // Pair factor 2 bounds the off-diagonal coupling; wall is fixed.
            float mobility=1/r.body[id].y+dot(localArm,localArm)*inverseInertia;
            float4 material=float4(r.c.kn,r.c.kt,r.c.mu,r.c.normalViscosity);
            if(wall)material=r.wallMaterial[key.y-r.c.N];
            float area=r.normalArea[contact].w*(wall?1:.5);
            float coupling=wall?1:2;
            stiffnessRate+=coupling*area*(material.x*(1+material.z)+material.y)*mobility;
            viscousRate+=coupling*area*material.w*(1+material.z)*mobility;
        }
        if(key.y>=r.c.N){
            compensated3(f,fLow,force);compensated3(t,tLow,cross(arm,force));
            stress0-=force.x*arm;stress1-=force.y*arm;stress2-=force.z*arm;
            energy+=r.contactForce[contact].w;denom+=length(force);
            maxDepth=max(maxDepth,r.pointDepth[contact].w/r.body[key.x].w);
            dissipation+=r.slip[contact].w;normalPower+=r.normalLossPower[contact];continue;
        }
        float3 branch=centerBranch(key.x,key.y);
        compensated3(f,fLow,source?force:-force);
        compensated3(t,tLow,source?cross(arm,force):cross(arm-branch,-force));
        // Compression-positive virial: -force_on_source outer centre branch.
        stress0-=.5*force.x*branch;stress1-=.5*force.y*branch;stress2-=.5*force.z*branch;
        energy+=.5*r.contactForce[contact].w;denom+=length(force);
        maxDepth=max(maxDepth,r.pointDepth[contact].w/min(r.body[key.x].w,r.body[key.y].w));
        dissipation+=.5*r.slip[contact].w;normalPower+=.5*r.normalLossPower[contact];
    }
    sharedF[lane]=float4(f,maxDepth);
    sharedFL[lane]=float4(fLow,0);
    sharedT[lane]=float4(t,dissipation);
    sharedTL[lane]=float4(tLow,0);
    sharedS0[lane]=float4(stress0,stiffnessRate);
    sharedS1[lane]=float4(stress1,viscousRate);
    sharedS2[lane]=float4(stress2,normalPower);
    sharedA[lane]=float4(energy,denom,0,0);
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=32;stride>0;stride/=2){
        if(lane<stride){
            float3 hf=sharedF[lane].xyz,lf=sharedFL[lane].xyz,ht=sharedT[lane].xyz,lt=sharedTL[lane].xyz;
            compensated3(hf,lf,sharedF[lane+stride].xyz);compensated3(hf,lf,sharedFL[lane+stride].xyz);
            compensated3(ht,lt,sharedT[lane+stride].xyz);compensated3(ht,lt,sharedTL[lane+stride].xyz);
            sharedF[lane]=float4(hf,max(sharedF[lane].w,sharedF[lane+stride].w));sharedFL[lane]=float4(lf,0);
            sharedT[lane]=float4(ht,sharedT[lane].w+sharedT[lane+stride].w);sharedTL[lane]=float4(lt,0);
            sharedS0[lane]+=sharedS0[lane+stride];
            sharedS1[lane]+=sharedS1[lane+stride];
            sharedS2[lane]+=sharedS2[lane+stride];
            sharedA[lane]+=sharedA[lane+stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(lane!=0)return;
    f=sharedF[0].xyz;fLow=sharedFL[0].xyz;t=sharedT[0].xyz;tLow=sharedTL[0].xyz;
    maxDepth=sharedF[0].w;dissipation=sharedT[0].w;
    stress0=sharedS0[0].xyz;stress1=sharedS1[0].xyz;stress2=sharedS2[0].xyz;
    stiffnessRate=sharedS0[0].w;viscousRate=sharedS1[0].w;normalPower=sharedS2[0].w;
    energy=sharedA[0].x;denom=sharedA[0].y;
    f=rounded3(f+fLow);t=rounded3(t+tLow);
    r.timestepRows[id]=float4(stiffnessRate,viscousRate,0,0);
    float4 normalState=r.normalDamping[id];normalState.z=normalPower;r.normalDamping[id]=normalState;
    r.force[id]=float4(f,maxDepth);r.torque[id]=float4(t,dissipation);
    r.particleStress[3*id]=float4(stress0,0);r.particleStress[3*id+1]=float4(stress1,0);r.particleStress[3*id+2]=float4(stress2,0);
    float kinetic=.5*(r.body[id].y*dot(r.velocity[id].xyz,r.velocity[id].xyz)+dot(r.momentum[id].xyz,omega(id)));
    r.particleDiagnostic[id]=float4(kinetic,energy,length(f),denom);
    accumulate(id,float4(dissipation,0,0,0));
}

// Separate dispatch makes a rejection visible before any particle starts a kick.
// Trial contacts are checked again before the final kick; host rollback is atomic.
[shader("compute")][numthreads(64,1,1)]
void checkContactTimestep(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0||r.c.dt==0)return;
    float dt=r.c.dt;float2 rates=r.timestepRows[id].xy;
    float3 v=r.velocity[id].xyz+r.velocityRemainder[id].xyz;
    float speed=length(v)+dt*length(r.force[id].xyz)/r.body[id].y;
    float inverseInertia=max(max(dot(abs(r.invI[3*id].xyz),float3(1)),dot(abs(r.invI[3*id+1].xyz),float3(1))),dot(abs(r.invI[3*id+2].xyz),float3(1)));
    // Bound spin through angular momentum, including torque-free anisotropy.
    float spin=inverseInertia*(length(r.momentum[id].xyz+r.momentumRemainder[id].xyz)+dt*length(r.torque[id].xyz));
    float wallSpeed=0;
    for(uint w=0;w<r.c.wallCount;w++)wallSpeed=max(wallSpeed,length(r.wallVelocity[w].xyz));
    // Prevent grossly underresolved flight, rotation and imposed cell strain too.
    float motion=dt*(speed+r.body[id].z*spin+wallSpeed);
    bool finite=all(isfinite(rates))&&isfinite(motion)&&isfinite(spin);
    if(!finite){fail(512);return;}
    if(dt*sqrt(rates.x)>.1||dt*rates.y>.1||dt*spin>.01||
       motion>.25*r.c.skin||any(abs(r.cell[1].xyz)*dt>.01))fail(8192);
}

// Each (wall,body) owns its rows; only integer atomics collect contacts.
// Reduction over bodies is performed by diagnostic.slang, without force atomics.
[shader("compute")][numthreads(64,1,1)]
void gatherWallRows(uint index:SV_DispatchThreadID){
    if(index>=r.c.N*r.c.wallCount||r.control[0]!=0)return;
    uint body=index%r.c.N,wall=index/r.c.N,count=0,visited=0;
    float3 reaction=0,moment=0;float power=0;uint entry=r.heads[body];
    while(entry!=EMPTY){
        if(visited++>2*r.c.contactCapacity){fail(128);return;}
        uint contact=entry/2;
        if(r.meta[contact].y==r.c.N+wall){
            float3 force=r.contactForce[contact].xyz;
            float3 point=r.position[body].xyz+r.positionRemainder[body].xyz+r.pointDepth[contact].xyz;
            reaction-=force;moment-=cross(point,force);
            power+=dot(force,r.wallVelocity[wall].xyz);count++;
        }
        entry=r.next[entry];
    }
    float4 work=r.wallAccumulated[index];
    r.wallRows[3*index]=float4(reaction,asfloat(count));
    r.wallRows[3*index+1]=float4(moment,power);
    r.wallRows[3*index+2]=float4(work.x+work.y,0,0,0);
}

[shader("compute")][numthreads(64,1,1)]
void beginStep(uint id:SV_DispatchThreadID){
    if(id>=r.c.N||r.control[0]!=0)return;
    float dt=r.c.dt;float3 rate=r.cell[1].xyz;
    accumulateNormalLoss(id,.5*dt);
    float3 v=r.velocity[id].xyz,vl=r.velocityRemainder[id].xyz;
    float3 l=r.momentum[id].xyz,ll=r.momentumRemainder[id].xyz;
    compensated3(v,vl,.5*dt*r.force[id].xyz/r.body[id].y);
    compensated3(l,ll,.5*dt*r.torque[id].xyz);
    float loss=-expm1Stable(-r.c.drag*dt)*(.5*r.body[id].y*dot(v+vl,v+vl)+rotationalEnergy(id,l+ll));
    accumulate(id,float4(0,loss,.5*dt*cellPower(id),wallStepWork(id,.5*dt)));
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
    accumulateNormalLoss(id,.5*r.c.dt);
    float3 v=r.velocity[id].xyz,vl=r.velocityRemainder[id].xyz;
    float3 l=r.momentum[id].xyz,ll=r.momentumRemainder[id].xyz;
    compensated3(v,vl,.5*r.c.dt*r.force[id].xyz/r.body[id].y);
    compensated3(l,ll,.5*r.c.dt*r.torque[id].xyz);
    float loss=-expm1Stable(-r.c.drag*r.c.dt)*(.5*r.body[id].y*dot(v+vl,v+vl)+rotationalEnergy(id,l+ll));
    accumulate(id,float4(0,loss,.5*r.c.dt*cellPower(id),wallStepWork(id,.5*r.c.dt)));
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
    r.diagnosticRows[id*8]=float4(r.particleStress[id*3].xyz,kinetic);
    r.diagnosticRows[id*8+1]=float4(r.particleStress[id*3+1].xyz,r.particleDiagnostic[id].y);
    r.diagnosticRows[id*8+2]=float4(r.particleStress[id*3+2].xyz,length(r.force[id].xyz));
    r.diagnosticRows[id*8+3]=float4(m*v.x*v,r.particleDiagnostic[id].w);
    r.diagnosticRows[id*8+4]=float4(m*v.y*v,r.force[id].w);
    r.diagnosticRows[id*8+5]=float4(m*v.z*v,r.torque[id].w);
    r.diagnosticRows[id*8+6]=r.accumulated[id];
    float4 normalState=r.normalDamping[id];
    r.diagnosticRows[id*8+7]=float4(normalState.x+normalState.y,normalState.z,0,0);
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

[shader("compute")][numthreads(64,1,1)]
void refreshWallPower(uint index:SV_DispatchThreadID){
    if(index>=r.c.N*r.c.wallCount)return;
    float4 value=r.wallRows[3*index+1];
    value.w=-dot(r.wallRows[3*index].xyz,r.wallVelocity[index/r.c.N].xyz);
    r.wallRows[3*index+1]=value;
}

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
    if(!denseGather())return;
    if(body>=r.c.N||r.control[0]!=0)return;
    uint count=r.adjCounts[body];
    if(count>8192){if(lane==0)fail(16384);return;}
    uint size=1;while(size<count)size*=2;
    uint offset=r.adjOffsets[body];uint lastMapped=EMPTY;
    if(lane==0)sortedEndpoints[8191]=r.adjCacheCounts[body]==count?0:1;
    GroupMemoryBarrierWithGroupSync();
    if(sortedEndpoints[8191]==0){
        for(uint i=lane;i<count;i+=64){
            uint4 cached=r.adjCacheKeys[8192*body+i];
            uint slot=hashKey(cached);uint found=EMPTY;
            for(uint probe=0;probe<r.c.hashSize;probe++){
                uint current=r.hash[slot];if(current==EMPTY)break;
                if(keyEqual(cached,r.meta[current])){found=current;break;}
                slot=(slot+1)&(r.c.hashSize-1);
            }
            if(found==EMPTY){uint unused;InterlockedOr(sortedEndpoints[8191],1,unused);}
            uint mapped=found==EMPTY?EMPTY:2*found+cached.w;
            if(i==8191)lastMapped=mapped;else sortedEndpoints[i]=mapped;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    bool reuse=sortedEndpoints[8191]==0;
    GroupMemoryBarrierWithGroupSync();
    if(reuse){
        for(uint i=lane;i<count;i+=64)r.adjEntries[offset+i]=i==8191?lastMapped:sortedEndpoints[i];
        if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[0],1,unused);}
        return;
    }
    if(lane==0){uint unused;InterlockedAdd(r.adjCacheMetrics[1],1,unused);}
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
    for(uint i=lane;i<count;i+=64){
        uint entry=sortedEndpoints[i];r.adjEntries[offset+i]=entry;
        r.adjCacheKeys[8192*body+i]=uint4(r.meta[entry/2].xyz,entry&1);
    }
    if(lane==0)r.adjCacheCounts[body]=count;
}
void gatherSparse(uint id){
    if(id>=r.c.N||r.control[0]!=0)return;
    float3 f=0,t=0,fLow=0,tLow=0;float3 stress0=0,stress1=0,stress2=0;float energy=0,denom=0,maxDepth=0,dissipation=0,normalPower=0;
    float stiffnessRate=0,viscousRate=0;
    float inverseInertia=max(max(dot(abs(r.invI[3*id].xyz),float3(1)),dot(abs(r.invI[3*id+1].xyz),float3(1))),dot(abs(r.invI[3*id+2].xyz),float3(1)));
    uint entry=r.heads[id];uint visited=0;
    while(entry!=EMPTY){
        if(visited++>2*r.c.contactCapacity){fail(128);return;}
        uint contact=entry/2;bool source=(entry&1)==0;uint4 key=r.meta[contact];
        float3 force=r.contactForce[contact].xyz;
        float3 arm=r.pointDepth[contact].xyz;
        if(r.c.activeTimestep!=0){
            bool wall=key.y>=r.c.N;
            float3 localArm=source?arm:arm-centerBranch(key.x,key.y);
            // Spectral bound for frozen contact geometry in mass-normalized 6D.
            // Coulomb coupling adds mu*kn (and mu*viscosity).
            // Pair factor 2 bounds the off-diagonal coupling; wall is fixed.
            float mobility=1/r.body[id].y+dot(localArm,localArm)*inverseInertia;
            float4 material=float4(r.c.kn,r.c.kt,r.c.mu,r.c.normalViscosity);
            if(wall)material=r.wallMaterial[key.y-r.c.N];
            float area=r.normalArea[contact].w*(wall?1:.5);
            float coupling=wall?1:2;
            stiffnessRate+=coupling*area*(material.x*(1+material.z)+material.y)*mobility;
            viscousRate+=coupling*area*material.w*(1+material.z)*mobility;
        }
        if(key.y>=r.c.N){
            compensated3(f,fLow,force);compensated3(t,tLow,cross(arm,force));
            stress0-=force.x*arm;stress1-=force.y*arm;stress2-=force.z*arm;
            energy+=r.contactForce[contact].w;denom+=length(force);
            maxDepth=max(maxDepth,r.pointDepth[contact].w/r.body[key.x].w);
            dissipation+=r.slip[contact].w;normalPower+=r.normalLossPower[contact];entry=r.next[entry];continue;
        }
        float3 branch=centerBranch(key.x,key.y);
        compensated3(f,fLow,source?force:-force);
        compensated3(t,tLow,source?cross(arm,force):cross(arm-branch,-force));
        // Compression-positive virial: -force_on_source outer centre branch.
        stress0-=.5*force.x*branch;stress1-=.5*force.y*branch;stress2-=.5*force.z*branch;
        energy+=.5*r.contactForce[contact].w;denom+=length(force);
        maxDepth=max(maxDepth,r.pointDepth[contact].w/min(r.body[key.x].w,r.body[key.y].w));
        dissipation+=.5*r.slip[contact].w;normalPower+=.5*r.normalLossPower[contact];
        entry=r.next[entry];
    }
    f=rounded3(f+fLow);t=rounded3(t+tLow);
    r.timestepRows[id]=float4(stiffnessRate,viscousRate,0,0);
    float4 normalState=r.normalDamping[id];normalState.z=normalPower;r.normalDamping[id]=normalState;
    r.force[id]=float4(f,maxDepth);r.torque[id]=float4(t,dissipation);
    r.particleStress[3*id]=float4(stress0,0);r.particleStress[3*id+1]=float4(stress1,0);r.particleStress[3*id+2]=float4(stress2,0);
    float kinetic=.5*(r.body[id].y*dot(r.velocity[id].xyz,r.velocity[id].xyz)+dot(r.momentum[id].xyz,omega(id)));
    r.particleDiagnostic[id]=float4(kinetic,energy,length(f),denom);
    accumulate(id,float4(dissipation,0,0,0));
}


[shader("compute")][numthreads(64,1,1)]
void gather(uint id:SV_DispatchThreadID,uint group:SV_GroupID,uint lane:SV_GroupThreadID){
    if(denseGather())gatherDense(group,lane);
    else gatherSparse(id);
}
