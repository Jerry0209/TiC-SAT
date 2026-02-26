#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params/L2Cache_Controller.hh"
#include "python/pybind11/core.hh"
#include "sim/init.hh"
#include "sim/sim_object.hh"

#include "mem/protocol/L2Cache_Controller.hh"

#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/structures/CacheMemory.hh"
#include "base/types.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "mem/ruby/network/MessageBuffer.hh"
#include "base/types.hh"
#include "mem/ruby/network/MessageBuffer.hh"
namespace py = pybind11;

static void
module_init(py::module &m_internal)
{
    py::module m = m_internal.def_submodule("param_L2Cache_Controller");
    py::class_<L2Cache_ControllerParams, RubyControllerParams, std::unique_ptr<L2Cache_ControllerParams, py::nodelete>>(m, "L2Cache_ControllerParams")
        .def(py::init<>())
        .def("create", &L2Cache_ControllerParams::create)
        .def_readwrite("GlobalRequestFromL2Cache", &L2Cache_ControllerParams::GlobalRequestFromL2Cache)
        .def_readwrite("GlobalRequestToL2Cache", &L2Cache_ControllerParams::GlobalRequestToL2Cache)
        .def_readwrite("L1RequestFromL2Cache", &L2Cache_ControllerParams::L1RequestFromL2Cache)
        .def_readwrite("L1RequestToL2Cache", &L2Cache_ControllerParams::L1RequestToL2Cache)
        .def_readwrite("L2cache", &L2Cache_ControllerParams::L2cache)
        .def_readwrite("request_latency", &L2Cache_ControllerParams::request_latency)
        .def_readwrite("responseFromL2Cache", &L2Cache_ControllerParams::responseFromL2Cache)
        .def_readwrite("responseToL2Cache", &L2Cache_ControllerParams::responseToL2Cache)
        .def_readwrite("response_latency", &L2Cache_ControllerParams::response_latency)
        .def_readwrite("triggerQueue", &L2Cache_ControllerParams::triggerQueue)
        ;

    py::class_<L2Cache_Controller, AbstractController, std::unique_ptr<L2Cache_Controller, py::nodelete>>(m, "L2Cache_Controller")
        ;

}

static EmbeddedPyBind embed_obj("L2Cache_Controller", module_init, "RubyController");
