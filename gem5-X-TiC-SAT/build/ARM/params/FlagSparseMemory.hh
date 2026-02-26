#ifndef __PARAMS__FlagSparseMemory__
#define __PARAMS__FlagSparseMemory__

class FlagSparseMemory;

#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "base/types.hh"

#include "params/BasicPioDevice.hh"

struct FlagSparseMemoryParams
    : public BasicPioDeviceParams
{
    FlagSparseMemory * create();
    Addr pio_addr;
    int32_t pio_size;
};

#endif // __PARAMS__FlagSparseMemory__
