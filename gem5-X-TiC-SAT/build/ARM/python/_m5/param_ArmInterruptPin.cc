#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmInterruptPin.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/base_gic.hh"

#include "base/types.hh"
#include "dev/platform.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmInterruptPin");
    py::class_<ArmInterruptPinParams, SimObjectParams, std::unique_ptr<ArmInterruptPinParams, py::nodelete>>(m, "ArmInterruptPinParams")
        .def_readwrite("num", &ArmInterruptPinParams::num)
        .def_readwrite("platform", &ArmInterruptPinParams::platform)
        ;

    py::class_<ArmInterruptPin, SimObject, std::unique_ptr<ArmInterruptPin, py::nodelete>>(m, "ArmInterruptPin")
        ;

}

static EmbeddedPyBind embed_obj("ArmInterruptPin", module_init, "SimObject");
