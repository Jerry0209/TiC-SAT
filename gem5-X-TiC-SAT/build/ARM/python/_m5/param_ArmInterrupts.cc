#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmInterrupts.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/interrupts.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmInterrupts");
    py::class_<ArmInterruptsParams, SimObjectParams, std::unique_ptr<ArmInterruptsParams, py::nodelete>>(m, "ArmInterruptsParams")
        .def(py::init<>())
        .def("create", &ArmInterruptsParams::create)
        ;

    py::class_<ArmISA::Interrupts, SimObject, std::unique_ptr<ArmISA::Interrupts, py::nodelete>>(m, "ArmISA_COLONS_Interrupts")
        ;

}

static EmbeddedPyBind embed_obj("ArmInterrupts", module_init, "SimObject");
