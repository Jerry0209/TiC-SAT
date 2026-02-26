#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Pl011.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/pl011.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Pl011");
    py::class_<Pl011Params, UartParams, std::unique_ptr<Pl011Params, py::nodelete>>(m, "Pl011Params")
        .def(py::init<>())
        .def("create", &Pl011Params::create)
        .def_readwrite("end_on_eot", &Pl011Params::end_on_eot)
        .def_readwrite("gic", &Pl011Params::gic)
        .def_readwrite("int_delay", &Pl011Params::int_delay)
        .def_readwrite("int_num", &Pl011Params::int_num)
        ;

    py::class_<Pl011, Uart, std::unique_ptr<Pl011, py::nodelete>>(m, "Pl011")
        ;

}

static EmbeddedPyBind embed_obj("Pl011", module_init, "Uart");
