#include <iostream>

#include "luaT.h"
#include "THC.h"

using namespace std;

namespace sparse_convo { namespace cuda {

int SparseFilterConvo::UpdateOutput(lua_State *L)
{
    cout << "Hello World from CUDA Land!" << endl;
    return 0;
}

int SparseFilterConvo::UpdateGradInput(lua_State *L)
{
    return 0;
}

int SparseFilterConvo::AccGradParameters(lua_State *L)
{
    return 0;
}


} }
