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

    int64_t kW = luaT_getfieldcheckint(L, 1, "m_kW");
    int64_t kH = luaT_getfieldcheckint(L, 1, "m_kH");
    int64_t dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    int64_t dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    
    const auto weight = get_mem_tensor(L, "weight");
    const auto bias   = get_mem_tensor(L, "bias");
    auto opProcMat = get_mem_tensor(L, "opProcMat");
    auto output = get_mem_tensor(L, "output");

    cout << "Update Output:" << endl
         << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
         << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

    luaL_argcheck(L, input->nDimension == 4, 2, "Only a 4D (batch mode) tensor is supported");
   
    const int64_t nOutputPlane = luaT_getfieldcheckint(L, 1, "m_nOutputPlanes");
    const int64_t nInputPlane = luaT_getfieldcheckint(L, 1, "m_nInputPlanes");

    cout << "Input Planes: " << nInputPlane << endl
         << "Output Planes: " << nOutputPlane << endl;

    const int64_t width = input->size[3];
    const int64_t height = input->size[2];

    cout << "Width: " << width << endl
         << "Height: " << height << endl;

    // NOTE: Padding is assumed, so input sizes equal output sizes,
    // except for the number of planes
    const int64_t batchSize = input->size[0];
    const int64_t chanSize = width * height;
    const int64_t inputImgSize = nInputPlane * chanSize;
    const int64_t outputImgSize = nOutputPlane * chanSize;

    cout << "Batch Size: " << batchSize << endl
         << "Channel Size: " << chanSize << endl
         << "Input Image Size: " << inputImgSize << endl
         << "Output Image Size: " << outputImgSize << endl;

    auto pOutputData = THFloatTensor_data(output);
    const auto pInputData = THFloatTensor_data(input);
    const auto pBiasData = THFloatTensor_data(bias);
    auto pProcData = THFloatTensor_data(opProcMat);

    int64_t i,j,k;

    // Initialize the output to the biases
    for (i = 0; i < batchSize; ++i)
    {
        #pragma omp parallel for private(j)
        for (j = 0; j < nOutputPlane; ++j)
        {
            auto pOutput = pOutputData + i * outputImgSize + j * chanSize;
            for (k = 0; k < chanSize; ++k)
            {
                pOutput[k] = pBiasData[j];
            }
        }
    }

    auto wvTensor = THFloatTensor_new();

    for (i = 0; i < batchSize; ++i)
    {
        const auto pInputImg = pInputData + i * inputImgSize;
        auto pOutputImg = pOutputData + i * outputImgSize;

        for (k = 0; k < nInputPlane; ++k)
        {
            // Get a matrix view of the weights for the current
            // input channel
            THFloatTensor_select(wvTensor, weights, 0, k);

            const auto pInputChan = pInputImg + k * chanSize;

            int64_t r, c, mR = 0;
            for (int64_t kR = -(kH / 2); kR <= (kH / 2); ++kR)
            {
                for (int64_t kC = -(kW / 2); kC < (kW / 2); ++kC, ++mR)
                {
                    #pragma omp parallel for private(r)
                    for (r = 0; r < height; ++r)
                    {
                        int64_t ipR = r + kR;

                        for (c = 0; c < width; ++c)
                        {
                            int64_t mC = r * width + c;

                            int64_t ipC = c + kC;

                            float val = 0.0f;
                            if (ipR >= 0 and ipC >= 0 and
                                ipR < height and ipC < width)
                            {
                                val = pInputChan[ipR * width + ipC];
                            }


                        }
                    }
                }
            }
        }
    }

    THFloatTensor_free(wvTensor);

    return 1;
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
