#ifndef __PARAMS__TarmacTracer__
#define __PARAMS__TarmacTracer__

namespace Trace {
class TarmacTracer;
} // namespace Trace

#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "base/types.hh"

#include "params/InstTracer.hh"

struct TarmacTracerParams
    : public InstTracerParams
{
    Trace::TarmacTracer * create();
    Tick end_tick;
    Tick start_tick;
};

#endif // __PARAMS__TarmacTracer__
