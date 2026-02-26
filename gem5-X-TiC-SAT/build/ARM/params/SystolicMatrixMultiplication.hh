#ifndef __PARAMS__SystolicMatrixMultiplication__
#define __PARAMS__SystolicMatrixMultiplication__

class SystolicMatrixMultiplication;

#include <vector>
#include "params/BaseCPU.hh"
#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "base/types.hh"

#include "params/BasicPioDevice.hh"

struct SystolicMatrixMultiplicationParams
    : public BasicPioDeviceParams
{
    SystolicMatrixMultiplication * create();
    std::vector< BaseCPU * > cpus;
    Addr pio_addr;
    int32_t pio_size;
};

#endif // __PARAMS__SystolicMatrixMultiplication__
