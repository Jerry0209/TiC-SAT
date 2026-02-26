#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/AmbaIntDevice.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/amba_device.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_AmbaIntDevice");
    py::class_<AmbaIntDeviceParams, AmbaPioDeviceParams, std::unique_ptr<AmbaIntDeviceParams, py::nodelete>>(m, "AmbaIntDeviceParams")
        .def_readwrite("gic", &AmbaIntDeviceParams::gic)
        .def_readwrite("int_delay", &AmbaIntDeviceParams::int_delay)
        .def_readwrite("int_num", &AmbaIntDeviceParams::int_num)
        ;

    py::class_<AmbaIntDevice, AmbaPioDevice, std::unique_ptr<AmbaIntDevice, py::nodelete>>(m, "AmbaIntDevice")
        ;

}

static EmbeddedPyBind embed_obj("AmbaIntDevice", module_init, "AmbaPioDevice");
