#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/LinuxArmSystem.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "arch/arm/linux/system.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_LinuxArmSystem");
    py::class_<LinuxArmSystemParams, GenericArmSystemParams, std::unique_ptr<LinuxArmSystemParams, py::nodelete>>(m, "LinuxArmSystemParams")
        .def(py::init<>())
        .def("create", &LinuxArmSystemParams::create)
        ;

    py::class_<LinuxArmSystem, GenericArmSystem, std::unique_ptr<LinuxArmSystem, py::nodelete>>(m, "LinuxArmSystem")
        .def("dumpDmesg", &LinuxArmSystem::dumpDmesg)
        ;

}

static EmbeddedPyBind embed_obj("LinuxArmSystem", module_init, "GenericArmSystem");
