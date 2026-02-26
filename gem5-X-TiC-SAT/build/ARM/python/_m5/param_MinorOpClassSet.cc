#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MinorOpClassSet.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/minor/func_unit.hh"

#include <vector>
#include "cpu/minor/func_unit.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MinorOpClassSet");
    py::class_<MinorOpClassSetParams, SimObjectParams, std::unique_ptr<MinorOpClassSetParams, py::nodelete>>(m, "MinorOpClassSetParams")
        .def(py::init<>())
        .def("create", &MinorOpClassSetParams::create)
        .def_readwrite("opClasses", &MinorOpClassSetParams::opClasses)
        ;

    py::class_<MinorOpClassSet, SimObject, std::unique_ptr<MinorOpClassSet, py::nodelete>>(m, "MinorOpClassSet")
        ;

}

static EmbeddedPyBind embed_obj("MinorOpClassSet", module_init, "SimObject");
