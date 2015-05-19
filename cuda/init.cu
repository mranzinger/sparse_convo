#include "luaT.h"
#include "THC.h"

#include "SparseFilterConvolution.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libsparseconvo(lua_State *L);

int luaopen_libsparseconvo(lua_State *L)
{
    lua_newtable(L);
    sc_SparseFilterConvo_init(L);

    return 1;
}
