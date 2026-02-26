#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/FreebsdArmSystem.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/freebsd/system.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_FreebsdArmSystem");
    py::class_<FreebsdArmSystemParams, GenericArmSystemParams, std::unique_ptr<FreebsdArmSystemParams, py::nodelete>>(m, "FreebsdArmSystemParams")
        .def(py::init<>())
        .def("create", &FreebsdArmSystemParams::create)
        ;

    py::class_<FreebsdArmSystem, GenericArmSystem, std::unique_ptr<FreebsdArmSystem, py::nodelete>>(m, "FreebsdArmSystem")
        ;

}

static EmbeddedPyBind embed_obj("FreebsdArmSystem", module_init, "GenericArmSystem");
