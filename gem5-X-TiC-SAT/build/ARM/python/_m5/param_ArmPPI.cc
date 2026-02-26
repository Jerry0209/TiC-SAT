#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmPPI.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/base_gic.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmPPI");
    py::class_<ArmPPIParams, ArmInterruptPinParams, std::unique_ptr<ArmPPIParams, py::nodelete>>(m, "ArmPPIParams")
        .def(py::init<>())
        .def("create", &ArmPPIParams::create)
        ;

    py::class_<ArmPPI, ArmInterruptPin, std::unique_ptr<ArmPPI, py::nodelete>>(m, "ArmPPI")
        ;

}

static EmbeddedPyBind embed_obj("ArmPPI", module_init, "ArmInterruptPin");
