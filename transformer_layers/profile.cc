#include "profile.h"

#include "run_mode_config.h"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {

#if CFG_GEM5_PROFILE_REGIONS
const char* kGem5ProfileIndexPath = "gem5_profile_regions.tsv";
#endif
std::size_t gem5_profile_dump_index = 0;

bool m5Available() {
    static int cached_m5_available = -1;
    if (cached_m5_available < 0) {
        cached_m5_available =
            (std::system("command -v m5 >/dev/null 2>&1") == 0) ? 1 : 0;
    }
    return cached_m5_available == 1;
}

void runM5IfAvailable(const char* command) {
    if (m5Available()) {
        std::system(command);
    }
}

#if CFG_GEM5_PROFILE_REGIONS
void dumpTransformerStatsCheckpoint(const char* checkpoint,
                                    const char* interval_since_previous) {
    gem5_profile_dump_index++;

    std::ofstream index(kGem5ProfileIndexPath, std::ios::app);
    if (index.is_open()) {
        index << gem5_profile_dump_index << "\t"
              << checkpoint << "\t"
              << interval_since_previous << "\n";
    }

    std::cout << "[GEM5_PROFILE] dump " << gem5_profile_dump_index
              << " checkpoint=" << checkpoint
              << " interval=" << interval_since_previous << std::endl;
    runM5IfAvailable("m5 dumpstats");
}
#endif

} // namespace

void resetTransformerStatsWindow(const char* scope) {
    gem5_profile_dump_index = 0;

#if CFG_GEM5_PROFILE_REGIONS
    std::ofstream index(kGem5ProfileIndexPath);
    if (index.is_open()) {
        index << "scope\t" << scope << "\n";
        index << "dump_index\tcheckpoint\tinterval_since_previous\n";
    }
#else
    (void)scope;
#endif

    std::cout << "[GEM5_PROFILE] reset scope=" << scope << std::endl;
    runM5IfAvailable("m5 resetstats");
}

void dumpTransformerStatsCheckpointIfProfiling(const char* checkpoint,
                                               const char* interval_since_previous) {
#if CFG_GEM5_PROFILE_REGIONS
    // Region profiling keeps cumulative snapshots.  Subtract adjacent dumps to
    // recover a per-region interval, while the last dump remains the total.
    dumpTransformerStatsCheckpoint(checkpoint, interval_since_previous);
#else
    (void)checkpoint;
    (void)interval_since_previous;
#endif
}

void dumpTransformerStatsLegacyBoundary(const char* checkpoint,
                                        const char* interval_since_previous) {
#if CFG_GEM5_PROFILE_REGIONS
    dumpTransformerStatsCheckpoint(checkpoint, interval_since_previous);
#else
    (void)checkpoint;
    (void)interval_since_previous;
    runM5IfAvailable("m5 dumpresetstats");
#endif
}
