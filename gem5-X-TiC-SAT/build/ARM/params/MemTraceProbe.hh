#ifndef __PARAMS__MemTraceProbe__
#define __PARAMS__MemTraceProbe__

class MemTraceProbe;

#include <cstddef>
#include "params/System.hh"
#include <cstddef>
#include <cstddef>
#include <string>
#include <cstddef>

#include "params/BaseMemProbe.hh"

struct MemTraceProbeParams
    : public BaseMemProbeParams
{
    MemTraceProbe * create();
    System * system;
    bool trace_compress;
    std::string trace_file;
    bool with_pc;
};

#endif // __PARAMS__MemTraceProbe__
