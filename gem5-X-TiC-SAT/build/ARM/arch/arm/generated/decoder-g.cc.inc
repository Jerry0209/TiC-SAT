// DO NOT EDIT
// This file was automatically generated from an ISA description:
//   arm/isa/main.isa


#include <string>

#include <gem5/asm/generic/m5ops.h>

#include "arch/arm/decoder.hh"
#include "arch/arm/faults.hh"
#include "arch/arm/intregs.hh"
#include "arch/arm/isa_traits.hh"
#include "arch/arm/utility.hh"
#include "base/loader/symtab.hh"
#include "base/cprintf.hh"
#include "cpu/thread_context.hh"

// ACM core integration
#include "arch/arm/system.hh"
#include "dev/arm/systolic_m2m.hh"
#include "dev/arm/flag_sparse_memory.hh"

using namespace ArmISA;
