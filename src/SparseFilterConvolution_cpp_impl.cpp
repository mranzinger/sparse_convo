#include <iostream>

using namespace std;

namespace sparse_convo { namespace cpu {

int SparseFilterConvo::UpdateOutput(lua_State *L)
{
    cout << "Hello World from CPU Land!" << endl;
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
