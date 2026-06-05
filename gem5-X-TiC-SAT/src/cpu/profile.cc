/*
 * Copyright (c) 2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Nathan Binkert
 */

#include "cpu/profile.hh"

#include <string>

#include "base/bitfield.hh"
#include "base/callback.hh"
#include "base/loader/symtab.hh"
#include "base/statistics.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "cpu/thread_context.hh"

using namespace std;

ProfileNode::ProfileNode()
    : count(0)
{ }

/**
 * Write this node and all children in gem5's function-profile dump format.
 *
 * @param symbol Printable function/scope name associated with this node.
 * @param id Stable identifier emitted for this node in the dump. Callers use
 *        the node address for child nodes and 0 for the synthetic top node.
 * @param symtab Symbol table used to translate child return addresses back to
 *        function names.
 * @param os Output stream that receives the profile graph.
 */
void
ProfileNode::dump(const string &symbol, uint64_t id, const SymbolTable *symtab,
                  ostream &os) const
{
    ccprintf(os, "%#x %s %d ", id, symbol, count);
    ChildList::const_iterator i, end = children.end();
    for (i = children.begin(); i != end; ++i) {
        const ProfileNode *node = i->second;
        ccprintf(os, "%#x ", (intptr_t)node);
    }

    ccprintf(os, "\n");

    for (i = children.begin(); i != end; ++i) {
        Addr addr = i->first;
        string symbol;
        if (addr == 1)
            symbol = "user";
        else if (addr == 2)
            symbol = "console";
        else if (addr == 3)
            symbol = "unknown";
        else if (!symtab->findSymbol(addr, symbol))
            panic("could not find symbol for address %#x\n", addr);

        const ProfileNode *node = i->second;
        node->dump(symbol, (intptr_t)node, symtab, os);
    }
}

/**
 * Reset this node's sample count and recursively reset all child nodes.
 */
void
ProfileNode::clear()
{
    count = 0;
    ChildList::iterator i, end = children.end();
    for (i = children.begin(); i != end; ++i)
        i->second->clear();
}

/**
 * Create a function profiler using the simulator's symbol table.
 *
 * @param _symtab Symbol table used later to translate sampled PCs and stack
 *        addresses into function names during dump() and sample().
 *
 * The constructor also registers a stats reset callback so gem5 can clear this
 * profile together with the rest of the simulator statistics.
 */
FunctionProfile::FunctionProfile(const SymbolTable *_symtab)
    : reset(0), symtab(_symtab)
{
    reset = new MakeCallback<FunctionProfile, &FunctionProfile::clear>(this);
    Stats::registerResetCallback(reset);
}

FunctionProfile::~FunctionProfile()
{
    if (reset)
        delete reset;
}

/**
 * Convert a sampled call stack into nodes in the profile tree.
 *
 * @param stack Vector of program counters collected by the ISA stack tracer.
 *        Entries are consumed in reverse vector order so the tree root follows
 *        the caller side of the stack and leaves represent the sampled frame.
 *
 * @return ProfileNode corresponding to the leaf frame for this stack sample.
 */
ProfileNode *
FunctionProfile::consume(const vector<Addr> &stack)
{
    ProfileNode *current = &top;
    for (int i = 0, size = stack.size(); i < size; ++i) {
        ProfileNode *&ptr = current->children[stack[size - i - 1]];
        if (ptr == NULL)
            ptr = new ProfileNode;

        current = ptr;
    }

    return current;
}

/**
 * Clear all accumulated function-tree and per-PC profile counters.
 */
void
FunctionProfile::clear()
{
    top.clear();
    pc_count.clear();
}

/**
 * Dump the flat PC histogram and hierarchical function profile.
 *
 * @param tc Thread context associated with the profile dump. The current
 *        implementation does not need it, but the parameter keeps the public
 *        interface aligned with CPU/profile callers.
 * @param os Output stream that receives the text profile data.
 */
void
FunctionProfile::dump(ThreadContext *tc, ostream &os) const
{
    (void)tc;

    ccprintf(os, ">>>PC data\n");
    map<Addr, Counter>::const_iterator i, end = pc_count.end();
    for (i = pc_count.begin(); i != end; ++i) {
        Addr pc = i->first;
        Counter count = i->second;

        std::string symbol;
        if (pc == 1)
            ccprintf(os, "user %d\n", count);
        else if (symtab->findSymbol(pc, symbol) && !symbol.empty())
            ccprintf(os, "%s %d\n", symbol, count);
        else
            ccprintf(os, "%#x %d\n", pc, count);
    }

    ccprintf(os, ">>>function data\n");
    top.dump("top", 0, symtab, os);
}

/**
 * Record one profiling sample for a leaf node and program counter.
 *
 * @param node Leaf ProfileNode returned by consume(); its count is incremented
 *        to represent one observed stack sample.
 * @param pc Current program counter for the sample. If the symbol table can map
 *        it to a function start address, that symbol address is counted;
 *        otherwise the raw PC is counted so unsymbolized samples are preserved.
 */
void
FunctionProfile::sample(ProfileNode *node, Addr pc)
{
    node->count++;

    Addr symaddr;
    if (symtab->findNearestAddr(pc, symaddr)) {
        pc_count[symaddr]++;
    } else {
        // record PC even if we don't have a symbol to avoid
        // silently biasing the histogram
        pc_count[pc]++;
    }
}
