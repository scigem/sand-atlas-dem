#include "normal.slang"

// Qualification operator for a fixed-normal translating penalty plane.
// Full one-sided surface traction: this is an explicit project wall variant.
// Reused by the Engine wall collector and the independent qualification probe.
struct PlaneResponse {
    float4 forceEnergy;
    float4 slipLoss;
    bool sliding;
    float normalLossPower;
};

PlaneResponse planeViscousResponse(float3 normal, float signedDistance, float area,
                            float3 relativeVelocity, float3 oldSlip,
                            float kn, float kt, float mu, float viscosity, float dt) {
    PlaneResponse value;
    value.forceEnergy=0; value.slipLoss=0; value.sliding=false; value.normalLossPower=0;
    if(signedDistance>=0)return value;
    float depth=-signedDistance;
    float3 tangent=relativeVelocity-normal*dot(relativeVelocity,normal);
    float3 trial=oldSlip-normal*dot(oldSlip,normal)+dt*tangent;
    float3 slip=trial;
    NormalResponse response=normalResponse(depth,area,dot(relativeVelocity,normal),kn,viscosity,1.);
    float fn=response.force;value.normalLossPower=response.lossPower;
    float3 ft=-kt*area*slip;
    float magnitude=length(ft);
    if(magnitude>mu*fn && magnitude>0){
        value.sliding=true;
        ft*=mu*fn/magnitude;
        slip=-ft/(kt*area);
    }
    value.forceEnergy=float4(fn*normal+ft,
        .5*area*(kn*depth*depth+kt*dot(slip,slip)));
    value.slipLoss=float4(slip,max(0.,-dot(ft,trial-slip)));
    return value;
}

// Retain the independently qualified zero-viscosity probe API.
PlaneResponse planeResponse(float3 normal,float signedDistance,float area,
                            float3 relativeVelocity,float3 oldSlip,
                            float kn,float kt,float mu,float dt){
    return planeViscousResponse(normal,signedDistance,area,relativeVelocity,oldSlip,kn,kt,mu,0,dt);
}

[shader("compute")][numthreads(64,1,1)]
void planeProbe(uint id:SV_DispatchThreadID, uniform uint N,
                uniform float3 normal, uniform float offset,
                uniform float kn, uniform float kt, uniform float mu, uniform float dt,
                StructuredBuffer<float4> pointArea,
                StructuredBuffer<float4> relativeVelocity,
                StructuredBuffer<float4> oldSlipActive,
                RWStructuredBuffer<float4> forceEnergy,
                RWStructuredBuffer<float4> slipLoss,
                RWStructuredBuffer<float4> torqueActive) {
    if(id>=N)return;
    float4 point=pointArea[id];
    float phi=dot(normal,point.xyz)-offset;
    float3 old=oldSlipActive[id].w!=0?oldSlipActive[id].xyz:float3(0);
    PlaneResponse value=planeResponse(normal,phi,point.w,relativeVelocity[id].xyz,
                                      old,kn,kt,mu,dt);
    forceEnergy[id]=value.forceEnergy;
    slipLoss[id]=value.slipLoss;
    torqueActive[id]=float4(cross(point.xyz,value.forceEnergy.xyz),phi<0?1.:0.);
}
