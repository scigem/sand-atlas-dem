// Study-only ABI bridge to frozen Slang-generated C++, not a second force law.
#include "planeProbe.cpp"

extern "C" void cpu_plane(uint32_t count, float* points, float* velocities,
                           float* slips, float* forces, float* losses, float* torques,
                           float kn, float kt, float mu, float dt, uint32_t repetitions) {
    using V = Vector<float, 4>;
    EntryPointParams_0 p = {};
    p.N_0 = count;
    p.normal_0 = {0, 0, 1};
    p.offset_0 = 0;
    p.kn_0 = kn; p.kt_0 = kt; p.mu_0 = mu; p.dt_0 = dt;
    p.pointArea_0 = {reinterpret_cast<V*>(points), count};
    p.relativeVelocity_0 = {reinterpret_cast<V*>(velocities), count};
    p.oldSlipActive_0 = {reinterpret_cast<V*>(slips), count};
    p.forceEnergy_1 = {reinterpret_cast<V*>(forces), count};
    p.slipLoss_1 = {reinterpret_cast<V*>(losses), count};
    p.torqueActive_0 = {reinterpret_cast<V*>(torques), count};
    ComputeVaryingInput varying = {};
    varying.endGroupID = {(count + 63) / 64, 1, 1};
    for (uint32_t i=0; i<repetitions; ++i) planeProbe(&varying, &p, nullptr);
}
