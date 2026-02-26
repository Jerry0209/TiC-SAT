#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/FlagSparseMemory.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/flag_sparse_memory.hh"

#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_FlagSparseMemory");
    py::class_<FlagSparseMemoryParams, BasicPioDeviceParams, std::unique_ptr<FlagSparseMemoryParams, py::nodelete>>(m, "FlagSparseMemoryParams")
        .def(py::init<>())
        .def("create", &FlagSparseMemoryParams::create)
        .def_readwrite("pio_addr", &FlagSparseMemoryParams::pio_addr)
        .def_readwrite("pio_size", &FlagSparseMemoryParams::pio_size)
        ;

    py::class_<FlagSparseMemory, BasicPioDevice, std::unique_ptr<FlagSparseMemory, py::nodelete>>(m, "FlagSparseMemory")
        ;

}

static EmbeddedPyBind embed_obj("FlagSparseMemory", module_init, "BasicPioDevice");
