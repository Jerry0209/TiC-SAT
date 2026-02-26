#ifndef __PARAMS__TrafficGen__
#define __PARAMS__TrafficGen__

class TrafficGen;

#include <cstddef>
#include <string>

#include "params/BaseTrafficGen.hh"

struct TrafficGenParams
    : public BaseTrafficGenParams
{
    TrafficGen * create();
    std::string config_file;
};

#endif // __PARAMS__TrafficGen__
