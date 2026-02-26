#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/GenericArmPciHost.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/pci_host.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "enums/ArmPciIntRouting.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_GenericArmPciHost");
    py::class_<GenericArmPciHostParams, GenericPciHostParams, std::unique_ptr<GenericArmPciHostParams, py::nodelete>>(m, "GenericArmPciHostParams")
        .def(py::init<>())
        .def("create", &GenericArmPciHostParams::create)
        .def_readwrite("int_base", &GenericArmPciHostParams::int_base)
        .def_readwrite("int_count", &GenericArmPciHostParams::int_count)
        .def_readwrite("int_policy", &GenericArmPciHostParams::int_policy)
        ;

    py::class_<GenericArmPciHost, GenericPciHost, std::unique_ptr<GenericArmPciHost, py::nodelete>>(m, "GenericArmPciHost")
        ;

}

static EmbeddedPyBind embed_obj("GenericArmPciHost", module_init, "GenericPciHost");
