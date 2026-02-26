#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MinorOpClass.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/minor/func_unit.hh"

#include "enums/OpClass.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MinorOpClass");
    py::class_<MinorOpClassParams, SimObjectParams, std::unique_ptr<MinorOpClassParams, py::nodelete>>(m, "MinorOpClassParams")
        .def(py::init<>())
        .def("create", &MinorOpClassParams::create)
        .def_readwrite("opClass", &MinorOpClassParams::opClass)
        ;

    py::class_<MinorOpClass, SimObject, std::unique_ptr<MinorOpClass, py::nodelete>>(m, "MinorOpClass")
        ;

}

static EmbeddedPyBind embed_obj("MinorOpClass", module_init, "SimObject");
