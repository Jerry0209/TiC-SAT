#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/GenericTimer.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/generic_timer.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "arch/arm/system.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_GenericTimer");
    py::class_<GenericTimerParams, ClockedObjectParams, std::unique_ptr<GenericTimerParams, py::nodelete>>(m, "GenericTimerParams")
        .def(py::init<>())
        .def("create", &GenericTimerParams::create)
        .def_readwrite("gic", &GenericTimerParams::gic)
        .def_readwrite("int_hyp", &GenericTimerParams::int_hyp)
        .def_readwrite("int_phys_ns", &GenericTimerParams::int_phys_ns)
        .def_readwrite("int_phys_s", &GenericTimerParams::int_phys_s)
        .def_readwrite("int_virt", &GenericTimerParams::int_virt)
        .def_readwrite("system", &GenericTimerParams::system)
        ;

    py::class_<GenericTimer, ClockedObject, std::unique_ptr<GenericTimer, py::nodelete>>(m, "GenericTimer")
        ;

}

static EmbeddedPyBind embed_obj("GenericTimer", module_init, "ClockedObject");
