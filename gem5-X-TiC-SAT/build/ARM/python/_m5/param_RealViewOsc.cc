#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/RealViewOsc.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/rv_ctrl.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "dev/arm/rv_ctrl.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "sim/voltage_domain.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_RealViewOsc");
    py::class_<RealViewOscParams, ClockDomainParams, std::unique_ptr<RealViewOscParams, py::nodelete>>(m, "RealViewOscParams")
        .def(py::init<>())
        .def("create", &RealViewOscParams::create)
        .def_readwrite("dcc", &RealViewOscParams::dcc)
        .def_readwrite("device", &RealViewOscParams::device)
        .def_readwrite("freq", &RealViewOscParams::freq)
        .def_readwrite("parent", &RealViewOscParams::parent)
        .def_readwrite("position", &RealViewOscParams::position)
        .def_readwrite("site", &RealViewOscParams::site)
        .def_readwrite("voltage_domain", &RealViewOscParams::voltage_domain)
        ;

    py::class_<RealViewOsc, ClockDomain, std::unique_ptr<RealViewOsc, py::nodelete>>(m, "RealViewOsc")
        ;

}

static EmbeddedPyBind embed_obj("RealViewOsc", module_init, "ClockDomain");
