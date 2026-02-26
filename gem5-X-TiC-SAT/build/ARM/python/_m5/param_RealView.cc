#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/RealView.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/realview.hh"

#include "sim/system.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_RealView");
    py::class_<RealViewParams, PlatformParams, std::unique_ptr<RealViewParams, py::nodelete>>(m, "RealViewParams")
        .def(py::init<>())
        .def("create", &RealViewParams::create)
        .def_readwrite("system", &RealViewParams::system)
        ;

    py::class_<RealView, Platform, std::unique_ptr<RealView, py::nodelete>>(m, "RealView")
        ;

}

static EmbeddedPyBind embed_obj("RealView", module_init, "Platform");
