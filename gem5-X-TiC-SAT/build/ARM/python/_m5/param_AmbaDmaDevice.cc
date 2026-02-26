#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/AmbaDmaDevice.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/amba_device.hh"

#include "base/types.hh"
#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_AmbaDmaDevice");
    py::class_<AmbaDmaDeviceParams, DmaDeviceParams, std::unique_ptr<AmbaDmaDeviceParams, py::nodelete>>(m, "AmbaDmaDeviceParams")
        .def_readwrite("amba_id", &AmbaDmaDeviceParams::amba_id)
        .def_readwrite("gic", &AmbaDmaDeviceParams::gic)
        .def_readwrite("int_num", &AmbaDmaDeviceParams::int_num)
        .def_readwrite("pio_addr", &AmbaDmaDeviceParams::pio_addr)
        .def_readwrite("pio_latency", &AmbaDmaDeviceParams::pio_latency)
        ;

    py::class_<AmbaDmaDevice, DmaDevice, std::unique_ptr<AmbaDmaDevice, py::nodelete>>(m, "AmbaDmaDevice")
        ;

}

static EmbeddedPyBind embed_obj("AmbaDmaDevice", module_init, "DmaDevice");
