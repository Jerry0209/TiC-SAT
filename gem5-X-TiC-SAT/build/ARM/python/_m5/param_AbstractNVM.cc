#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/AbstractNVM.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/abstract_nvm.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_AbstractNVM");
    py::class_<AbstractNVMParams, SimObjectParams, std::unique_ptr<AbstractNVMParams, py::nodelete>>(m, "AbstractNVMParams")
        ;

    py::class_<AbstractNVM, SimObject, std::unique_ptr<AbstractNVM, py::nodelete>>(m, "AbstractNVM")
        ;

}

static EmbeddedPyBind embed_obj("AbstractNVM", module_init, "SimObject");
