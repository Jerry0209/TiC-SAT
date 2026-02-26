#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Pl390.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/gic_pl390.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Pl390");
    py::class_<Pl390Params, BaseGicParams, std::unique_ptr<Pl390Params, py::nodelete>>(m, "Pl390Params")
        .def(py::init<>())
        .def("create", &Pl390Params::create)
        .def_readwrite("cpu_addr", &Pl390Params::cpu_addr)
        .def_readwrite("cpu_pio_delay", &Pl390Params::cpu_pio_delay)
        .def_readwrite("dist_addr", &Pl390Params::dist_addr)
        .def_readwrite("dist_pio_delay", &Pl390Params::dist_pio_delay)
        .def_readwrite("gem5_extensions", &Pl390Params::gem5_extensions)
        .def_readwrite("int_latency", &Pl390Params::int_latency)
        .def_readwrite("it_lines", &Pl390Params::it_lines)
        ;

    py::class_<Pl390, BaseGic, std::unique_ptr<Pl390, py::nodelete>>(m, "Pl390")
        ;

}

static EmbeddedPyBind embed_obj("Pl390", module_init, "BaseGic");
