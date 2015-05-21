
#include "SparseFilterConvolution_cpu_impl.h"

using namespace sparse_convo;

static int sparseconvo_SparseFilterConvo_cpu_updateOutput(lua_State *L)
{
    return cpu::SparseFilterConvo::UpdateOutput(L);
}

static int sparseconvo_SparseFilterConvo_cpu_updateGradInput(lua_State *L)
{
    return cpu::SparseFilterConvo::UpdateGradInput(L);
}

static int sparseconvo_SparseFilterConvo_cpu_accGradParameters(lua_State *L)
{
    return cpu::SparseFilterConvo::AccGradParameters(L);
}


static const struct luaL_Reg sparseconvo_SparseFilterConvo_cpu__ [] = {
    {"SparseFilterConvo_cpu_updateOutput", sparseconvo_SparseFilterConvo_cpu_updateOutput},
    {"SparseFilterConvo_cpu_updateGradInput", sparseconvo_SparseFilterConvo_cpu_updateGradInput},
    {"SparseFilterConvo_cpu_accGradParameters", sparseconvo_SparseFilterConvo_cpu_accGradParameters},
    {NULL, NULL}  
}; 

static void sc_SparseFilterConvo_cpu_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.FloatTensor");
    luaT_registeratname(L, sparseconvo_SparseFilterConvo_cpu__, "nn");
    lua_pop(L,1);
}
