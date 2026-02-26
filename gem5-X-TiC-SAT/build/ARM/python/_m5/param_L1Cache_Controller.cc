#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/L1Cache_Controller.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "mem/protocol/L1Cache_Controller.hh"

#include "mem/ruby/structures/CacheMemory.hh"
#include "mem/ruby/structures/CacheMemory.hh"
#include "base/types.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "base/types.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/system/Sequencer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "base/types.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_L1Cache_Controller");
    py::class_<L1Cache_ControllerParams, RubyControllerParams, std::unique_ptr<L1Cache_ControllerParams, py::nodelete>>(m, "L1Cache_ControllerParams")
        .def(py::init<>())
        .def("create", &L1Cache_ControllerParams::create)
        .def_readwrite("L1Dcache", &L1Cache_ControllerParams::L1Dcache)
        .def_readwrite("L1Icache", &L1Cache_ControllerParams::L1Icache)
        .def_readwrite("l2_select_num_bits", &L1Cache_ControllerParams::l2_select_num_bits)
        .def_readwrite("mandatoryQueue", &L1Cache_ControllerParams::mandatoryQueue)
        .def_readwrite("requestFromL1Cache", &L1Cache_ControllerParams::requestFromL1Cache)
        .def_readwrite("requestToL1Cache", &L1Cache_ControllerParams::requestToL1Cache)
        .def_readwrite("request_latency", &L1Cache_ControllerParams::request_latency)
        .def_readwrite("responseFromL1Cache", &L1Cache_ControllerParams::responseFromL1Cache)
        .def_readwrite("responseToL1Cache", &L1Cache_ControllerParams::responseToL1Cache)
        .def_readwrite("send_evictions", &L1Cache_ControllerParams::send_evictions)
        .def_readwrite("sequencer", &L1Cache_ControllerParams::sequencer)
        .def_readwrite("triggerQueue", &L1Cache_ControllerParams::triggerQueue)
        .def_readwrite("use_timeout_latency", &L1Cache_ControllerParams::use_timeout_latency)
        ;

    py::class_<L1Cache_Controller, AbstractController, std::unique_ptr<L1Cache_Controller, py::nodelete>>(m, "L1Cache_Controller")
        ;

}

static EmbeddedPyBind embed_obj("L1Cache_Controller", module_init, "RubyController");
