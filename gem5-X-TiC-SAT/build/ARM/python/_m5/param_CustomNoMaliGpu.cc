#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/CustomNoMaliGpu.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/gpu_nomali.hh"

#include "base/types.hh"
#include "base/types.hh"
#include <vector>
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include <vector>
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_CustomNoMaliGpu");
    py::class_<CustomNoMaliGpuParams, NoMaliGpuParams, std::unique_ptr<CustomNoMaliGpuParams, py::nodelete>>(m, "CustomNoMaliGpuParams")
        .def(py::init<>())
        .def("create", &CustomNoMaliGpuParams::create)
        .def_readwrite("as_present", &CustomNoMaliGpuParams::as_present)
        .def_readwrite("gpu_id", &CustomNoMaliGpuParams::gpu_id)
        .def_readwrite("js_features", &CustomNoMaliGpuParams::js_features)
        .def_readwrite("js_present", &CustomNoMaliGpuParams::js_present)
        .def_readwrite("l2_features", &CustomNoMaliGpuParams::l2_features)
        .def_readwrite("l2_present", &CustomNoMaliGpuParams::l2_present)
        .def_readwrite("mem_features", &CustomNoMaliGpuParams::mem_features)
        .def_readwrite("mmu_features", &CustomNoMaliGpuParams::mmu_features)
        .def_readwrite("shader_present", &CustomNoMaliGpuParams::shader_present)
        .def_readwrite("texture_features", &CustomNoMaliGpuParams::texture_features)
        .def_readwrite("thread_features", &CustomNoMaliGpuParams::thread_features)
        .def_readwrite("thread_max_barrier_size", &CustomNoMaliGpuParams::thread_max_barrier_size)
        .def_readwrite("thread_max_threads", &CustomNoMaliGpuParams::thread_max_threads)
        .def_readwrite("thread_max_workgroup_size", &CustomNoMaliGpuParams::thread_max_workgroup_size)
        .def_readwrite("tiler_features", &CustomNoMaliGpuParams::tiler_features)
        .def_readwrite("tiler_present", &CustomNoMaliGpuParams::tiler_present)
        ;

    py::class_<CustomNoMaliGpu, NoMaliGpu, std::unique_ptr<CustomNoMaliGpu, py::nodelete>>(m, "CustomNoMaliGpu")
        ;

}

static EmbeddedPyBind embed_obj("CustomNoMaliGpu", module_init, "NoMaliGpu");
