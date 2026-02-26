#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/FlashDevice.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/flash_device.hh"

#include "base/types.hh"
#include "base/types.hh"
#include "enums/DataDistribution.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_FlashDevice");
    py::class_<FlashDeviceParams, AbstractNVMParams, std::unique_ptr<FlashDeviceParams, py::nodelete>>(m, "FlashDeviceParams")
        .def(py::init<>())
        .def("create", &FlashDeviceParams::create)
        .def_readwrite("GC_active", &FlashDeviceParams::GC_active)
        .def_readwrite("blk_size", &FlashDeviceParams::blk_size)
        .def_readwrite("data_distribution", &FlashDeviceParams::data_distribution)
        .def_readwrite("erase_lat", &FlashDeviceParams::erase_lat)
        .def_readwrite("num_planes", &FlashDeviceParams::num_planes)
        .def_readwrite("page_size", &FlashDeviceParams::page_size)
        .def_readwrite("read_lat", &FlashDeviceParams::read_lat)
        .def_readwrite("write_lat", &FlashDeviceParams::write_lat)
        ;

    py::class_<FlashDevice, AbstractNVM, std::unique_ptr<FlashDevice, py::nodelete>>(m, "FlashDevice")
        ;

}

static EmbeddedPyBind embed_obj("FlashDevice", module_init, "AbstractNVM");
