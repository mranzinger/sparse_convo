
#include <iostream>

using namespace std;

static int sparseconvo_SparseFilterConvo_updateOutput(lua_State *L)
{
    cout << "Hello World!" << endl;
    return 1;
}

static int sparseconvo_SparseFilterConvo_updateGradInput(lua_State *L)
{
    return 1;
}

static int sparseconvo_SparseFilterConvo_accGradParameters(lua_State *L)
{
    return 1;
}


static const struct luaL_Reg sparseconvo_SparseFilterConvo__ [] = {
    {"SparseFilterConvo_updateOutput", sparseconvo_SparseFilterConvo_updateOutput},
    {"SparseFilterConvo_updateGradInput", sparseconvo_SparseFilterConvo_updateGradInput},
    {"SparseFilterConvo_accGradParameters", sparseconvo_SparseFilterConvo_accGradParameters},
    {NULL, NULL}  
}; 

static void sc_SparseFilterConvo_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, sparseconvo_SparseFilterConvo__, "nn");
    lua_pop(L,1);
}
