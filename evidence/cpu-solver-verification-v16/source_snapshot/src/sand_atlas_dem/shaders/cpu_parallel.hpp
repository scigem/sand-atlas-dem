// Per-NativeKernel scratch: owned and freed through the C ABI, never process-global.
// Only capacities are reused. Every contact and count is reconstructed on dispatch.
struct Scratch {
        using Meta = std::remove_pointer_t<decltype(std::declval<@RES_TYPE@>().@META@.data)>;
        using Point = std::remove_pointer_t<decltype(std::declval<@RES_TYPE@>().@POINT@.data)>;
        using Normal = std::remove_pointer_t<decltype(std::declval<@RES_TYPE@>().@NORMAL@.data)>;
        struct Hits {
            std::vector<Meta> meta;
            std::vector<Point> point;
            std::vector<Normal> normal;
            uint32_t flags = 0, count = 0, attempts = 0;
        };
    std::vector<Hits> hits;
    std::mutex mutex;
    uint64_t stats[4] = {}; // pairs, query attempts, overflow retries, retained bytes
};
extern "C" void* cpu_scratch_create() noexcept {
    try { return new Scratch; } catch (...) { return nullptr; }
}
extern "C" void cpu_scratch_destroy(void* ptr) noexcept { delete static_cast<Scratch*>(ptr); }
extern "C" void cpu_scratch_stats(void* ptr, uint64_t* out) noexcept {
    auto& scratch = *static_cast<Scratch*>(ptr);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    std::memcpy(out, scratch.stats, sizeof(scratch.stats));
}
// @DISPATCH@
// Inserted inside the generated C ABI bridge, after compiler-defined resource
// binding. Each pair gets private output/control storage. Geometry and state
// are read-only; contactCounts has one writer per pair. Compaction is serial in
// pair order, preserving the original collector's arithmetic and contact order.
if (uint32_t(values[@THREAD_INDEX@]) > 1 && count > 1 && res.@CONTROL@.data[0] == 0) {
    try {
        auto& scratch = *static_cast<Scratch*>(scratch_ptr);
        std::lock_guard<std::mutex> lock(scratch.mutex);
        auto& hits = scratch.hits;
        hits.resize(count);
        std::memset(scratch.stats, 0, sizeof(scratch.stats));
        std::atomic<uint32_t> next{0};
        const uint32_t limit = res.@CONFIG@.@CAPACITY@;
        auto worker = [&]() {
            for (;;) {
                uint32_t pair = next.fetch_add(1, std::memory_order_relaxed);
                if (pair >= count) return;
                auto& hit = hits[pair];
                hit.flags = hit.count = hit.attempts = 0;
                uint32_t capacity = uint32_t(std::min(uint64_t(limit),
                    std::max(uint64_t(1024), uint64_t(hit.meta.size()))));
                for (;;) {
                    // Never shrink a warm allocation when the global limit changes.
                    // Grow independently so a caught allocation failure cannot leave
                    // one successful allocation hiding another unfinished allocation.
                    hit.meta.resize(std::max(size_t(capacity), hit.meta.size()));
                    hit.point.resize(std::max(size_t(capacity), hit.point.size()));
                    hit.normal.resize(std::max(size_t(capacity), hit.normal.size()));
                    ++hit.attempts;
                    auto local = res;
                    uint32_t control[16];
                    std::memcpy(control, res.@CONTROL@.data, sizeof(control));
                    control[2] = 0;
                    local.@CONTROL@.data = control;
                    local.@CONTROL@.count = 16;
                    local.@CONFIG@.@CAPACITY@ = capacity;
                    local.@META@.data = hit.meta.data(); local.@META@.count = capacity;
                    local.@POINT@.data = hit.point.data(); local.@POINT@.count = capacity;
                    local.@NORMAL@.data = hit.normal.data(); local.@NORMAL@.count = capacity;
                    res.@COUNTS@.data[pair] = 0;
                    auto localGlobals = globals;
                    localGlobals.@RESOURCE@ = &local;
                    auto localVi = vi;
                    localVi.startGroupID = {pair, 0, 0};
                    localVi.endGroupID = {pair + 1, 1, 1};
                    @ENTRY@(&localVi, @PARAMS@, &localGlobals);
                    if (control[0] == 32 && capacity < limit) {
                        capacity = uint32_t(std::min(uint64_t(limit), uint64_t(capacity) * 2));
                        continue;
                    }
                    hit.flags = control[0]; hit.count = control[2];
                    // Preserve allocated storage; only the valid prefix is copied.
                    break;
                }
            }
        };
        std::vector<std::future<void>> tasks;
        uint32_t workers = std::min(uint32_t(values[@THREAD_INDEX@]), count);
        for (uint32_t i = 0; i < workers; ++i)
            tasks.emplace_back(std::async(std::launch::async, worker));
        for (auto& task : tasks) task.get();
        uint64_t total = 0;
        uint32_t flags = 0;
        for (auto& hit : hits) {
            total += hit.count; flags |= hit.flags;
            scratch.stats[0]++;
            scratch.stats[1] += hit.attempts;
            scratch.stats[2] += hit.attempts - 1;
            scratch.stats[3] += hit.meta.capacity() * sizeof(Scratch::Meta)
                + hit.point.capacity() * sizeof(Scratch::Point)
                + hit.normal.capacity() * sizeof(Scratch::Normal);
        }
        res.@CONTROL@.data[2] = uint32_t(std::min(total, uint64_t(UINT32_MAX)));
        if (total > limit) flags |= 32;
        res.@CONTROL@.data[0] |= flags;
        if (flags) return 0; // Existing checked rollback/resize handles failure.
        size_t offset = 0;
        for (auto& hit : hits) {
            if (hit.count) {
                std::memcpy(res.@META@.data + offset, hit.meta.data(), hit.count * sizeof(Scratch::Meta));
                std::memcpy(res.@POINT@.data + offset, hit.point.data(), hit.count * sizeof(Scratch::Point));
                std::memcpy(res.@NORMAL@.data + offset, hit.normal.data(), hit.count * sizeof(Scratch::Normal));
            }
            offset += hit.count;
        }
        return 0;
    } catch (...) {
        // All async futures join during unwinding; no C++ exception crosses ctypes.
        return 1;
    }
}
