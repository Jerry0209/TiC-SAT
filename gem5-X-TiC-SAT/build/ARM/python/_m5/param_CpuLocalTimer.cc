#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/CpuLocalTimer.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/timer_cpulocal.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_CpuLocalTimer");
    py::class_<CpuLocalTimerParams, BasicPioDeviceParams, std::unique_ptr<CpuLocalTimerParams, py::nodelete>>(m, "CpuLocalTimerParams")
        .def(py::init<>())
        .def("create", &CpuLocalTimerParams::create)
        .def_readwrite("gic", &CpuLocalTimerParams::gic)
        .def_readwrite("int_num_timer", &CpuLocalTimerParams::int_num_timer)
        .def_readwrite("int_num_watchdog", &CpuLocalTimerParams::int_num_watchdog)
        ;

    py::class_<CpuLocalTimer, BasicPioDevice, std::unique_ptr<CpuLocalTimer, py::nodelete>>(m, "CpuLocalTimer")
        ;

}

static EmbeddedPyBind embed_obj("CpuLocalTimer", module_init, "BasicPioDevice");
