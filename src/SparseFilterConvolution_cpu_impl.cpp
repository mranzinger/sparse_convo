#include <iostream>

#include "luaT.h"
#include "TH.h"
#include "THC.h"


using namespace std;

namespace sparse_convo { namespace cpu {

inline THFloatTensor *get_mem_tensor(lua_State *L, const char *a_name, int a_idx = 1)
{
    return static_cast<THFloatTensor*>(
        luaT_getfieldcheckudata(L, a_idx, a_name, "torch.FloatTensor")
    );
}

int SparseFilterConvo::UpdateOutput(lua_State *L)
{
    auto input = (const THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");

    int kW = luaT_getfieldcheckint(L, 1, "m_kW");
    int kH = luaT_getfieldcheckint(L, 1, "m_kH");
    int dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    int dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    
    const auto weight = get_mem_tensor(L, "weight");
    const auto bias   = get_mem_tensor(L, "bias");
    auto output = get_mem_tensor(L, "output");

    cout << "Update Output:" << endl
         << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
         << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

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
