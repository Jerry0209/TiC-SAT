#ifndef __PARAMS__ArmSPI__
#define __PARAMS__ArmSPI__

class ArmSPI;


#include "params/ArmInterruptPin.hh"

struct ArmSPIParams
    : public ArmInterruptPinParams
{
    ArmSPI * create();
};

#endif // __PARAMS__ArmSPI__
