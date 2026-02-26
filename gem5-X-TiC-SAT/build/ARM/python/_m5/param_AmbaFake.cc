#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/AmbaFake.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/amba_fake.hh"

namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_AmbaFake");
    py::class_<AmbaFakeParams, AmbaPioDeviceParams, std::unique_ptr<AmbaFakeParams, py::nodelete>>(m, "AmbaFakeParams")
        .def(py::init<>())
        .def("create", &AmbaFakeParams::create)
        .def_readwrite("ignore_access", &AmbaFakeParams::ignore_access)
        ;

    py::class_<AmbaFake, AmbaPioDevice, std::unique_ptr<AmbaFake, py::nodelete>>(m, "AmbaFake")
        ;

}

static EmbeddedPyBind embed_obj("AmbaFake", module_init, "AmbaPioDevice");
