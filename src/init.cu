#include "luaT.h"
#include "THC.h"

#include "SparseFilterConvolution.cu"
#include "SparseFilterConvolution.cpp"

#include "SparseFilterConvolution_cu_impl.cu"
#include "SparseFilterConvolution_cpu_impl.cpp"

LUA_EXTERNC DLL_EXPORT int luaopen_libsparseconvo(lua_State *L);

int luaopen_libsparseconvo(lua_State *L)
{
    lua_newtable(L);
    sc_SparseFilterConvo_cu_init(L);
    sc_SparseFilterConvo_cpu_init(L);

    return 1;
}
