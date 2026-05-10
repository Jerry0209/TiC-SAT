#pragma once

void resetTransformerStatsWindow(const char* scope);

void dumpTransformerStatsCheckpointIfProfiling(const char* checkpoint,
                                               const char* interval_since_previous);

void dumpTransformerStatsLegacyBoundary(const char* checkpoint,
                                        const char* interval_since_previous);
