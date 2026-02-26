#ifndef __PARAMS__BaseTrafficGen__
#define __PARAMS__BaseTrafficGen__

class BaseTrafficGen;

#include <cstddef>
#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "params/System.hh"

#include "params/MemObject.hh"

struct BaseTrafficGenParams
    : public MemObjectParams
{
    bool elastic_req;
    Tick progress_check;
    System * system;
    unsigned int port_port_connection_count;
};

#endif // __PARAMS__BaseTrafficGen__
