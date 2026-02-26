#ifndef __PARAMS__Pl050__
#define __PARAMS__Pl050__

class Pl050;

#include <cstddef>
#include "params/PS2Device.hh"

#include "params/AmbaIntDevice.hh"

struct Pl050Params
    : public AmbaIntDeviceParams
{
    Pl050 * create();
    PS2Device * ps2;
};

#endif // __PARAMS__Pl050__
