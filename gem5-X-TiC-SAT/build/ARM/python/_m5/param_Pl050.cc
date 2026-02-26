#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Pl050.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/kmi.hh"

#include "dev/ps2/device.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Pl050");
    py::class_<Pl050Params, AmbaIntDeviceParams, std::unique_ptr<Pl050Params, py::nodelete>>(m, "Pl050Params")
        .def(py::init<>())
        .def("create", &Pl050Params::create)
        .def_readwrite("ps2", &Pl050Params::ps2)
        ;

    py::class_<Pl050, AmbaIntDevice, std::unique_ptr<Pl050, py::nodelete>>(m, "Pl050")
        ;

}

static EmbeddedPyBind embed_obj("Pl050", module_init, "AmbaIntDevice");
