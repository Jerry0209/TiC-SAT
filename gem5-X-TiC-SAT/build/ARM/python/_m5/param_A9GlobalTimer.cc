#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/A9GlobalTimer.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/timer_a9global.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_A9GlobalTimer");
    py::class_<A9GlobalTimerParams, BasicPioDeviceParams, std::unique_ptr<A9GlobalTimerParams, py::nodelete>>(m, "A9GlobalTimerParams")
        .def(py::init<>())
        .def("create", &A9GlobalTimerParams::create)
        .def_readwrite("gic", &A9GlobalTimerParams::gic)
        .def_readwrite("int_num", &A9GlobalTimerParams::int_num)
        ;

    py::class_<A9GlobalTimer, BasicPioDevice, std::unique_ptr<A9GlobalTimer, py::nodelete>>(m, "A9GlobalTimer")
        ;

}

static EmbeddedPyBind embed_obj("A9GlobalTimer", module_init, "BasicPioDevice");
