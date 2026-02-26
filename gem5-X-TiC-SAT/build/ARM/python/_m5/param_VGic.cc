#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/VGic.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/vgic.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "dev/platform.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_VGic");
    py::class_<VGicParams, PioDeviceParams, std::unique_ptr<VGicParams, py::nodelete>>(m, "VGicParams")
        .def(py::init<>())
        .def("create", &VGicParams::create)
        .def_readwrite("gic", &VGicParams::gic)
        .def_readwrite("hv_addr", &VGicParams::hv_addr)
        .def_readwrite("pio_delay", &VGicParams::pio_delay)
        .def_readwrite("platform", &VGicParams::platform)
        .def_readwrite("ppint", &VGicParams::ppint)
        .def_readwrite("vcpu_addr", &VGicParams::vcpu_addr)
        ;

    py::class_<VGic, PioDevice, std::unique_ptr<VGic, py::nodelete>>(m, "VGic")
        ;

}

static EmbeddedPyBind embed_obj("VGic", module_init, "PioDevice");
