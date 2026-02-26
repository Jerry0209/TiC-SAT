#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Sp804.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/timer_sp804.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Sp804");
    py::class_<Sp804Params, AmbaPioDeviceParams, std::unique_ptr<Sp804Params, py::nodelete>>(m, "Sp804Params")
        .def(py::init<>())
        .def("create", &Sp804Params::create)
        .def_readwrite("clock0", &Sp804Params::clock0)
        .def_readwrite("clock1", &Sp804Params::clock1)
        .def_readwrite("gic", &Sp804Params::gic)
        .def_readwrite("int_num0", &Sp804Params::int_num0)
        .def_readwrite("int_num1", &Sp804Params::int_num1)
        ;

    py::class_<Sp804, AmbaPioDevice, std::unique_ptr<Sp804, py::nodelete>>(m, "Sp804")
        ;

}

static EmbeddedPyBind embed_obj("Sp804", module_init, "AmbaPioDevice");
