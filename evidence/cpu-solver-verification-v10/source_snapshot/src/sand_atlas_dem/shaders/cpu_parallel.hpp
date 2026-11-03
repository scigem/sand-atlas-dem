// Inserted inside the generated C ABI bridge, after compiler-defined resource
// binding. Each pair gets private output/control storage. Geometry and state
// are read-only; contactCounts has one writer per pair. Compaction is serial in
// pair order, preserving the original collector's arithmetic and contact order.
if (uint32_t(values[@THREAD_INDEX@]) > 1 && count > 1 && res.@CONTROL@.data[0] == 0) {
    try {
        using Meta = std::remove_pointer_t<decltype(res.@META@.data)>;
        using Point = std::remove_pointer_t<decltype(res.@POINT@.data)>;
        using Normal = std::remove_pointer_t<decltype(res.@NORMAL@.data)>;
        struct Hits {
            std::vector<Meta> meta;
            std::vector<Point> point;
            std::vector<Normal> normal;
            uint32_t flags = 0, count = 0;
        };
        std::vector<Hits> hits(count);
        std::atomic<uint32_t> next{0};
        const uint32_t limit = res.@CONFIG@.@CAPACITY@;
        auto worker = [&]() {
            for (;;) {
                uint32_t pair = next.fetch_add(1, std::memory_order_relaxed);
                if (pair >= count) return;
                auto& hit = hits[pair];
                uint32_t capacity = std::min(uint32_t(1024), limit);
                for (;;) {
                    hit.meta.resize(capacity);
                    hit.point.resize(capacity);
                    hit.normal.resize(capacity);
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
        for (auto& hit : hits) { total += hit.count; flags |= hit.flags; }
        res.@CONTROL@.data[2] = uint32_t(std::min(total, uint64_t(UINT32_MAX)));
        if (total > limit) flags |= 32;
        res.@CONTROL@.data[0] |= flags;
        if (flags) return 0; // Existing checked rollback/resize handles failure.
        size_t offset = 0;
        for (auto& hit : hits) {
            if (hit.count) {
                std::memcpy(res.@META@.data + offset, hit.meta.data(), hit.count * sizeof(Meta));
                std::memcpy(res.@POINT@.data + offset, hit.point.data(), hit.count * sizeof(Point));
                std::memcpy(res.@NORMAL@.data + offset, hit.normal.data(), hit.count * sizeof(Normal));
            }
            offset += hit.count;
        }
        return 0;
    } catch (...) {
        // All async futures join during unwinding; no C++ exception crosses ctypes.
        return 1;
    }
}
