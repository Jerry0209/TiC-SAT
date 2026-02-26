#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/Gicv2mFrame.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/gic_v2m.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_Gicv2mFrame");
    py::class_<Gicv2mFrameParams, SimObjectParams, std::unique_ptr<Gicv2mFrameParams, py::nodelete>>(m, "Gicv2mFrameParams")
        .def(py::init<>())
        .def("create", &Gicv2mFrameParams::create)
        .def_readwrite("addr", &Gicv2mFrameParams::addr)
        .def_readwrite("spi_base", &Gicv2mFrameParams::spi_base)
        .def_readwrite("spi_len", &Gicv2mFrameParams::spi_len)
        ;

    py::class_<Gicv2mFrame, SimObject, std::unique_ptr<Gicv2mFrame, py::nodelete>>(m, "Gicv2mFrame")
        ;

}

static EmbeddedPyBind embed_obj("Gicv2mFrame", module_init, "SimObject");
