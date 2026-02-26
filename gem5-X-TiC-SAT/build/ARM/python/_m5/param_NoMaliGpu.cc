#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/NoMaliGpu.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/gpu_nomali.hh"

#include "enums/NoMaliGpuType.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "dev/arm/realview.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_NoMaliGpu");
    py::class_<NoMaliGpuParams, PioDeviceParams, std::unique_ptr<NoMaliGpuParams, py::nodelete>>(m, "NoMaliGpuParams")
        .def(py::init<>())
        .def("create", &NoMaliGpuParams::create)
        .def_readwrite("gpu_type", &NoMaliGpuParams::gpu_type)
        .def_readwrite("int_gpu", &NoMaliGpuParams::int_gpu)
        .def_readwrite("int_job", &NoMaliGpuParams::int_job)
        .def_readwrite("int_mmu", &NoMaliGpuParams::int_mmu)
        .def_readwrite("pio_addr", &NoMaliGpuParams::pio_addr)
        .def_readwrite("platform", &NoMaliGpuParams::platform)
        .def_readwrite("ver_maj", &NoMaliGpuParams::ver_maj)
        .def_readwrite("ver_min", &NoMaliGpuParams::ver_min)
        .def_readwrite("ver_status", &NoMaliGpuParams::ver_status)
        ;

    py::class_<NoMaliGpu, PioDevice, std::unique_ptr<NoMaliGpu, py::nodelete>>(m, "NoMaliGpu")
        ;

}

static EmbeddedPyBind embed_obj("NoMaliGpu", module_init, "PioDevice");
