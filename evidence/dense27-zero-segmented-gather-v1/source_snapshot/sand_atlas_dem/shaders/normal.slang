// Project normal traction viscosity. No conversion from restitution or nodal mass.
struct NormalResponse {float force;float lossPower;};
NormalResponse normalResponse(float depth,float area,float normalVelocity,
                              float stiffness,float viscosity,float weight){
    NormalResponse value;value.force=0;value.lossPower=0;
    if(depth<=0)return value;
    float elastic=weight*stiffness*area*depth;
    value.force=viscosity==0?elastic:max(0.,elastic-weight*viscosity*area*normalVelocity);
    value.lossPower=max(0.,(elastic-value.force)*normalVelocity);
    return value;
}
[shader("compute")][numthreads(64,1,1)]
void normalProbe(uint id:SV_DispatchThreadID,uniform uint N,
                 uniform float stiffness,uniform float viscosity,uniform float weight,
                 StructuredBuffer<float4> input,RWStructuredBuffer<float4> output){
    if(id>=N)return;
    float4 data=input[id];
    NormalResponse value=normalResponse(data.x,data.y,data.z,stiffness,viscosity,weight);
    output[id]=float4(value.force,value.lossPower,0,0);
}
