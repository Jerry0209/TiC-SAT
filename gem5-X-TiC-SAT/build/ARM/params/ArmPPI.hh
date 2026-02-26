#ifndef __PARAMS__ArmPPI__
#define __PARAMS__ArmPPI__

class ArmPPI;


#include "params/ArmInterruptPin.hh"

struct ArmPPIParams
    : public ArmInterruptPinParams
{
    ArmPPI * create();
};

#endif // __PARAMS__ArmPPI__
