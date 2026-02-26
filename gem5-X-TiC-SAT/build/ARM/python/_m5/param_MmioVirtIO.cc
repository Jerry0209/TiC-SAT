#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/MmioVirtIO.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/vio_mmio.hh"

#include "dev/arm/base_gic.hh"
#include "base/types.hh"
#include "dev/virtio/base.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_MmioVirtIO");
    py::class_<MmioVirtIOParams, BasicPioDeviceParams, std::unique_ptr<MmioVirtIOParams, py::nodelete>>(m, "MmioVirtIOParams")
        .def(py::init<>())
        .def("create", &MmioVirtIOParams::create)
        .def_readwrite("interrupt", &MmioVirtIOParams::interrupt)
        .def_readwrite("pio_size", &MmioVirtIOParams::pio_size)
        .def_readwrite("vio", &MmioVirtIOParams::vio)
        ;

    py::class_<MmioVirtIO, BasicPioDevice, std::unique_ptr<MmioVirtIO, py::nodelete>>(m, "MmioVirtIO")
        ;

}

static EmbeddedPyBind embed_obj("MmioVirtIO", module_init, "BasicPioDevice");
