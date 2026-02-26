#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/ArmSPI.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/base_gic.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_ArmSPI");
    py::class_<ArmSPIParams, ArmInterruptPinParams, std::unique_ptr<ArmSPIParams, py::nodelete>>(m, "ArmSPIParams")
        .def(py::init<>())
        .def("create", &ArmSPIParams::create)
        ;

    py::class_<ArmSPI, ArmInterruptPin, std::unique_ptr<ArmSPI, py::nodelete>>(m, "ArmSPI")
        ;

}

static EmbeddedPyBind embed_obj("ArmSPI", module_init, "ArmInterruptPin");
