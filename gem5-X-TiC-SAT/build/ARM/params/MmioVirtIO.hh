#ifndef __PARAMS__MmioVirtIO__
#define __PARAMS__MmioVirtIO__

class MmioVirtIO;

#include <cstddef>
#include "params/ArmInterruptPin.hh"
#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "params/VirtIODeviceBase.hh"

#include "params/BasicPioDevice.hh"

struct MmioVirtIOParams
    : public BasicPioDeviceParams
{
    MmioVirtIO * create();
    ArmInterruptPin * interrupt;
    Addr pio_size;
    VirtIODeviceBase * vio;
};

#endif // __PARAMS__MmioVirtIO__
