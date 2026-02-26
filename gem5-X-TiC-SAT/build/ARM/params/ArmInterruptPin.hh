#ifndef __PARAMS__ArmInterruptPin__
#define __PARAMS__ArmInterruptPin__

class ArmInterruptPin;

#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "params/Platform.hh"

#include "params/SimObject.hh"

struct ArmInterruptPinParams
    : public SimObjectParams
{
    uint32_t num;
    Platform * platform;
};

#endif // __PARAMS__ArmInterruptPin__
