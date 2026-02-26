#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MinorFUPool.hh"
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
    py::module m = m_internal.def_submodule("param_MinorFUPool");
    py::class_<MinorFUPoolParams, SimObjectParams, std::unique_ptr<MinorFUPoolParams, py::nodelete>>(m, "MinorFUPoolParams")
        .def(py::init<>())
        .def("create", &MinorFUPoolParams::create)
        .def_readwrite("funcUnits", &MinorFUPoolParams::funcUnits)
        ;

    py::class_<MinorFUPool, SimObject, std::unique_ptr<MinorFUPool, py::nodelete>>(m, "MinorFUPool")
        ;

}

static EmbeddedPyBind embed_obj("MinorFUPool", module_init, "SimObject");
