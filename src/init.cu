#include "luaT.h"
#include "THC.h"

#include "SparseFilterConvolution.cu"
#include "SparseFilterConvolution.cpp"

LUA_EXTERNC DLL_EXPORT int luaopen_libsparseconvo(lua_State *L);

int luaopen_libsparseconvo(lua_State *L)
{
    lua_newtable(L);
    sc_SparseFilterConvo_cu_init(L);
    sc_SparseFilterConvo_cpp_init(L);

    return 1;
}
