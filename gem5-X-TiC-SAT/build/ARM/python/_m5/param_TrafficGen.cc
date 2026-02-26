#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/TrafficGen.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/testers/traffic_gen/traffic_gen.hh"

#include <string>
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_TrafficGen");
    py::class_<TrafficGenParams, BaseTrafficGenParams, std::unique_ptr<TrafficGenParams, py::nodelete>>(m, "TrafficGenParams")
        .def(py::init<>())
        .def("create", &TrafficGenParams::create)
        .def_readwrite("config_file", &TrafficGenParams::config_file)
        ;

    py::class_<TrafficGen, BaseTrafficGen, std::unique_ptr<TrafficGen, py::nodelete>>(m, "TrafficGen")
        ;

}

static EmbeddedPyBind embed_obj("TrafficGen", module_init, "BaseTrafficGen");
