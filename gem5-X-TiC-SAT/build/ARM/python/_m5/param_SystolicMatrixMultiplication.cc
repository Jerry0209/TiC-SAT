#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/SystolicMatrixMultiplication.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "dev/arm/systolic_m2m.hh"

#include <vector>
#include "cpu/base.hh"
#include "base/types.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_SystolicMatrixMultiplication");
    py::class_<SystolicMatrixMultiplicationParams, BasicPioDeviceParams, std::unique_ptr<SystolicMatrixMultiplicationParams, py::nodelete>>(m, "SystolicMatrixMultiplicationParams")
        .def(py::init<>())
        .def("create", &SystolicMatrixMultiplicationParams::create)
        .def_readwrite("cpus", &SystolicMatrixMultiplicationParams::cpus)
        .def_readwrite("pio_addr", &SystolicMatrixMultiplicationParams::pio_addr)
        .def_readwrite("pio_size", &SystolicMatrixMultiplicationParams::pio_size)
        ;

    py::class_<SystolicMatrixMultiplication, BasicPioDevice, std::unique_ptr<SystolicMatrixMultiplication, py::nodelete>>(m, "SystolicMatrixMultiplication")
        ;

}

static EmbeddedPyBind embed_obj("SystolicMatrixMultiplication", module_init, "BasicPioDevice");
