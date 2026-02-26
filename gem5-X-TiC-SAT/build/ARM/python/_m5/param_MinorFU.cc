#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MinorFU.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "cpu/minor/func_unit.hh"

#include <vector>
#include "base/types.hh"
#include "base/types.hh"
#include "cpu/minor/func_unit.hh"
#include "base/types.hh"
#include <vector>
#include "cpu/minor/func_unit.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MinorFU");
    py::class_<MinorFUParams, SimObjectParams, std::unique_ptr<MinorFUParams, py::nodelete>>(m, "MinorFUParams")
        .def(py::init<>())
        .def("create", &MinorFUParams::create)
        .def_readwrite("cantForwardFromFUIndices", &MinorFUParams::cantForwardFromFUIndices)
        .def_readwrite("issueLat", &MinorFUParams::issueLat)
        .def_readwrite("opClasses", &MinorFUParams::opClasses)
        .def_readwrite("opLat", &MinorFUParams::opLat)
        .def_readwrite("timings", &MinorFUParams::timings)
        ;

    py::class_<MinorFU, SimObject, std::unique_ptr<MinorFU, py::nodelete>>(m, "MinorFU")
        ;

}

static EmbeddedPyBind embed_obj("MinorFU", module_init, "SimObject");
