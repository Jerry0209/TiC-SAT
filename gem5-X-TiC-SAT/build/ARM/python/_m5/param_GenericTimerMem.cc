#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/GenericTimerMem.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/generic_timer.hh"

#include "base/types.hh"
#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_GenericTimerMem");
    py::class_<GenericTimerMemParams, PioDeviceParams, std::unique_ptr<GenericTimerMemParams, py::nodelete>>(m, "GenericTimerMemParams")
        .def(py::init<>())
        .def("create", &GenericTimerMemParams::create)
        .def_readwrite("base", &GenericTimerMemParams::base)
        .def_readwrite("gic", &GenericTimerMemParams::gic)
        .def_readwrite("int_phys", &GenericTimerMemParams::int_phys)
        .def_readwrite("int_virt", &GenericTimerMemParams::int_virt)
        ;

    py::class_<GenericTimerMem, PioDevice, std::unique_ptr<GenericTimerMem, py::nodelete>>(m, "GenericTimerMem")
        ;

}

static EmbeddedPyBind embed_obj("GenericTimerMem", module_init, "PioDevice");
