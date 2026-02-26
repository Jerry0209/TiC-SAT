#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/UFSHostDevice.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/ufs_device.hh"

#include "dev/arm/base_gic.hh"
#include <vector>
#include "dev/storage/disk_image.hh"
#include "base/types.hh"
#include "base/types.hh"
#include <vector>
#include "dev/arm/abstract_nvm.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_UFSHostDevice");
    py::class_<UFSHostDeviceParams, DmaDeviceParams, std::unique_ptr<UFSHostDeviceParams, py::nodelete>>(m, "UFSHostDeviceParams")
        .def(py::init<>())
        .def("create", &UFSHostDeviceParams::create)
        .def_readwrite("gic", &UFSHostDeviceParams::gic)
        .def_readwrite("image", &UFSHostDeviceParams::image)
        .def_readwrite("img_blk_size", &UFSHostDeviceParams::img_blk_size)
        .def_readwrite("int_num", &UFSHostDeviceParams::int_num)
        .def_readwrite("internalflash", &UFSHostDeviceParams::internalflash)
        .def_readwrite("pio_addr", &UFSHostDeviceParams::pio_addr)
        .def_readwrite("pio_latency", &UFSHostDeviceParams::pio_latency)
        .def_readwrite("ufs_slots", &UFSHostDeviceParams::ufs_slots)
        ;

    py::class_<UFSHostDevice, DmaDevice, std::unique_ptr<UFSHostDevice, py::nodelete>>(m, "UFSHostDevice")
        ;

}

static EmbeddedPyBind embed_obj("UFSHostDevice", module_init, "DmaDevice");
