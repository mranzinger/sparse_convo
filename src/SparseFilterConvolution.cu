
#include "SparseFilterConvolution_cu_impl.cuh"

using namespace sparse_convo;

static int sparseconvo_SparseFilterConvo_cu_updateOutput(lua_State *L)
{
    return cuda::SparseFilterConvo::UpdateOutput(L);
}

static int sparseconvo_SparseFilterConvo_cu_updateGradInput(lua_State *L)
{
    return cuda::SparseFilterConvo::UpdateGradInput(L);
}

static int sparseconvo_SparseFilterConvo_cu_accGradParameters(lua_State *L)
{
    return cuda::SparseFilterConvo::AccGradParameters(L);
}


static const struct luaL_Reg sparseconvo_SparseFilterConvo_cu__ [] = {
    {"SparseFilterConvo_cu_updateOutput", sparseconvo_SparseFilterConvo_cu_updateOutput},
    {"SparseFilterConvo_cu_updateGradInput", sparseconvo_SparseFilterConvo_cu_updateGradInput},
    {"SparseFilterConvo_cu_accGradParameters", sparseconvo_SparseFilterConvo_cu_accGradParameters},
    {NULL, NULL}  
}; 

static void sc_SparseFilterConvo_cu_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, sparseconvo_SparseFilterConvo_cu__, "nn");
    lua_pop(L,1);
}
