#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/RealViewTemperatureSensor.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/rv_ctrl.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "dev/arm/rv_ctrl.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "sim/system.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_RealViewTemperatureSensor");
    py::class_<RealViewTemperatureSensorParams, SimObjectParams, std::unique_ptr<RealViewTemperatureSensorParams, py::nodelete>>(m, "RealViewTemperatureSensorParams")
        .def(py::init<>())
        .def("create", &RealViewTemperatureSensorParams::create)
        .def_readwrite("dcc", &RealViewTemperatureSensorParams::dcc)
        .def_readwrite("device", &RealViewTemperatureSensorParams::device)
        .def_readwrite("parent", &RealViewTemperatureSensorParams::parent)
        .def_readwrite("position", &RealViewTemperatureSensorParams::position)
        .def_readwrite("site", &RealViewTemperatureSensorParams::site)
        .def_readwrite("system", &RealViewTemperatureSensorParams::system)
        ;

    py::class_<RealViewTemperatureSensor, SimObject, std::unique_ptr<RealViewTemperatureSensor, py::nodelete>>(m, "RealViewTemperatureSensor")
        ;

}

static EmbeddedPyBind embed_obj("RealViewTemperatureSensor", module_init, "SimObject");
