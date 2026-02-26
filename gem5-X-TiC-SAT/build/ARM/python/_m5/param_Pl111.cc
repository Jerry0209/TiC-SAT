#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Pl111.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/pl111.hh"

#include "base/types.hh"
#include "base/vnc/vncinput.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Pl111");
    py::class_<Pl111Params, AmbaDmaDeviceParams, std::unique_ptr<Pl111Params, py::nodelete>>(m, "Pl111Params")
        .def(py::init<>())
        .def("create", &Pl111Params::create)
        .def_readwrite("enable_capture", &Pl111Params::enable_capture)
        .def_readwrite("pixel_clock", &Pl111Params::pixel_clock)
        .def_readwrite("vnc", &Pl111Params::vnc)
        ;

    py::class_<Pl111, AmbaDmaDevice, std::unique_ptr<Pl111, py::nodelete>>(m, "Pl111")
        ;

}

static EmbeddedPyBind embed_obj("Pl111", module_init, "AmbaDmaDevice");
