
struct Config {uint N;float factor;};
struct Resources {Config c;StructuredBuffer<float> input;RWStructuredBuffer<float> output;};
ParameterBlock<Resources> r;
[shader("compute")][numthreads(64,1,1)]
void apply(uint id:SV_DispatchThreadID,uniform float offset){
    if(id<r.c.N)r.output[id]=r.c.factor*r.input[id]+offset;
}
