
#include "SparseFilterConvolution_cpp_impl.h"

using namespace sparse_convo;

static int sparseconvo_SparseFilterConvo_cpp_updateOutput(lua_State *L)
{
    return cpu::SparseFilterConvo::UpdateOutput(L);
}

static int sparseconvo_SparseFilterConvo_cpp_updateGradInput(lua_State *L)
{
    return cpu::SparseFilterConvo::UpdateGradInput(L);
}

static int sparseconvo_SparseFilterConvo_cpp_accGradParameters(lua_State *L)
{
    return cpu::SparseFilterConvo::AccGradParameters(L);
}


static const struct luaL_Reg sparseconvo_SparseFilterConvo_cpp__ [] = {
    {"SparseFilterConvo_cpp_updateOutput", sparseconvo_SparseFilterConvo_cpp_updateOutput},
    {"SparseFilterConvo_cpp_updateGradInput", sparseconvo_SparseFilterConvo_cpp_updateGradInput},
    {"SparseFilterConvo_cpp_accGradParameters", sparseconvo_SparseFilterConvo_cpp_accGradParameters},
    {NULL, NULL}  
}; 

static void sc_SparseFilterConvo_cpp_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.FloatTensor");
    luaT_registeratname(L, sparseconvo_SparseFilterConvo_cpp__, "nn");
    lua_pop(L,1);
}
