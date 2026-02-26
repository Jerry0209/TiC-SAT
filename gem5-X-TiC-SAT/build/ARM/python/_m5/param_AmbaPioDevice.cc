#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/AmbaPioDevice.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/amba_device.hh"

#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_AmbaPioDevice");
    py::class_<AmbaPioDeviceParams, BasicPioDeviceParams, std::unique_ptr<AmbaPioDeviceParams, py::nodelete>>(m, "AmbaPioDeviceParams")
        .def_readwrite("amba_id", &AmbaPioDeviceParams::amba_id)
        ;

    py::class_<AmbaPioDevice, BasicPioDevice, std::unique_ptr<AmbaPioDevice, py::nodelete>>(m, "AmbaPioDevice")
        ;

}

static EmbeddedPyBind embed_obj("AmbaPioDevice", module_init, "BasicPioDevice");
