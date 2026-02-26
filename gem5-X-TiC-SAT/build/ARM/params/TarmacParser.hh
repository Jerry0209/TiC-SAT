#ifndef __PARAMS__TarmacParser__
#define __PARAMS__TarmacParser__

namespace Trace {
class TarmacParser;
} // namespace Trace

#include <cstddef>
#include <cstddef>
#include <cstddef>
#include <cstddef>
#include "base/types.hh"
#include "base/addr_range.hh"
#include <cstddef>
#include <cstddef>
#include <string>
#include <cstddef>
#include "base/types.hh"

#include "params/InstTracer.hh"

struct TarmacParserParams
    : public InstTracerParams
{
    Trace::TarmacParser * create();
    bool cpu_id;
    bool exit_on_diff;
    bool exit_on_insn_diff;
    AddrRange ignore_mem_addr;
    bool mem_wr_check;
    std::string path_to_trace;
    int start_pc;
};

#endif // __PARAMS__TarmacParser__
