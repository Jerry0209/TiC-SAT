#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/A9SCU.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/a9scu.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_A9SCU");
    py::class_<A9SCUParams, BasicPioDeviceParams, std::unique_ptr<A9SCUParams, py::nodelete>>(m, "A9SCUParams")
        .def(py::init<>())
        .def("create", &A9SCUParams::create)
        ;

    py::class_<A9SCU, BasicPioDevice, std::unique_ptr<A9SCU, py::nodelete>>(m, "A9SCU")
        ;

}

static EmbeddedPyBind embed_obj("A9SCU", module_init, "BasicPioDevice");
