#include <iostream>
#include <assert.h>
#include <sstream>

#include <omp.h>

#include "luaT.h"
#include "TH.h"
#include "THC.h"


using namespace std;

namespace sparse_convo { namespace cpu {

template<typename T>
struct tensor_trait;

template<>
struct tensor_trait<float>
{
    static const char name[];
    typedef THFloatTensor *th_type;
};
const char tensor_trait<float>::name[] = "torch.FloatTensor";

template<>
struct tensor_trait<int>
{
    static const char name[];
    typedef THIntTensor *th_type;
};
const char tensor_trait<int>::name[] = "torch.IntTensor";

template<typename T>
inline typename tensor_trait<T>::th_type get_mem_tensor(lua_State *L, const char *a_name, int a_idx = 1)
{
    return static_cast<typename tensor_trait<T>::th_type>(
        luaT_getfieldcheckudata(L, a_idx, a_name, tensor_trait<T>::name)
    );
}

int SparseFilterConvo::UpdateOutput(lua_State *L)
{
    auto input = (const THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");

    // Filter x Input Plane x Num Samples
    const auto weights = get_mem_tensor<float>(L, "weight");
    const auto bias   = get_mem_tensor<float>(L, "bias");
    auto output = get_mem_tensor<float>(L, "output");
    auto sampleOffsets = get_mem_tensor<int>(L, "sampleOffsets");

    //cout << "Update Output:" << endl
    //     << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
    //     << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

    //assert(THFloatTensor_nDimension(opProcMat) == 2);

    //cout << "OP Proc Mat: ["
    //     << THFloatTensor_size(opProcMat, 0) << " x "
    //     << THFloatTensor_size(opProcMat, 1) << "]"
    //     << endl;

    luaL_argcheck(L, input->nDimension == 4, 2, "Only a 4D (batch mode) tensor is supported");
   
    const int64_t nOutputPlane = luaT_getfieldcheckint(L, 1, "m_nOutputPlanes");
    const int64_t nInputPlane = luaT_getfieldcheckint(L, 1, "m_nInputPlanes");
    const int64_t nSamples = luaT_getfieldcheckint(L, 1, "m_numSamples");

    //cout << "Input Planes: " << nInputPlane << endl
    //     << "Output Planes: " << nOutputPlane << endl;

    const int64_t width = input->size[3];
    const int64_t height = input->size[2];

    //cout << "Width: " << width << endl
    //     << "Height: " << height << endl;

    // NOTE: Padding is assumed, so input sizes equal output sizes,
    // except for the number of planes
    const int64_t batchSize = input->size[0];
    const int64_t chanSize = width * height;
    const int64_t inputImgSize = nInputPlane * chanSize;
    const int64_t outputImgSize = nOutputPlane * chanSize;

    //cout << "Batch Size: " << batchSize << endl
    //     << "Channel Size: " << chanSize << endl
    //     << "Input Image Size: " << inputImgSize << endl
    //     << "Output Image Size: " << outputImgSize << endl;

    float * pOutputData = THFloatTensor_data(output);
    const float * pInputData = THFloatTensor_data(input);
    const float * pWeightData = THFloatTensor_data(weights);
    const float * pBiasData = THFloatTensor_data(bias);
    const int * pOffsetData = THIntTensor_data(sampleOffsets);

    int64_t i,j,k;

    //cout << "Filling Biases" << endl;

    // Initialize the output to the biases
    for (i = 0; i < batchSize; ++i)
    {
        for (j = 0; j < nOutputPlane; ++j)
        {
            auto pOutput = pOutputData + i * outputImgSize + j * chanSize;
            for (k = 0; k < chanSize; ++k)
            {
                pOutput[k] = pBiasData[j];
            }
        }
    }

    #pragma omp parallel for
    for (int64_t imageIdx = 0; imageIdx < batchSize; ++imageIdx)
    {
        for (int64_t outputIdx = 0; outputIdx < nOutputPlane; ++outputIdx)
        {
            for (int64_t sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
            {
                const int64_t ySampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2];
                const int64_t xSampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2 + 1];

                // We can compute the ranges where this specific part of the filter is valid
                const int64_t yStart = max<int64_t>(0, -ySampleOff);
                const int64_t yEnd = min(height, height - ySampleOff);

                const int64_t xStart = max<int64_t>(0, -xSampleOff);
                const int64_t xEnd = min(width, width - xSampleOff);

                for (int64_t inputIdx = 0; inputIdx < nInputPlane; ++inputIdx)
                {
                    const float sampleWeight = pWeightData[outputIdx * nInputPlane * nSamples + inputIdx * nSamples + sampleIdx];
                   
                    const float * pInputPlane = pInputData + imageIdx * inputImgSize + inputIdx * chanSize;
                    float * pOutputPlane = pOutputData + imageIdx * outputImgSize + outputIdx * chanSize;

                    for (int64_t y = yStart; y < yEnd; ++y)
                    {
                        const int64_t sY = y + ySampleOff;

                        const float * pInputRow = pInputPlane + sY * width;
                        float * pOutputRow = pOutputPlane + y * width;

                        for (int64_t x = xStart; x < xEnd; ++x)
                        {
                            const int64_t sX = x + xSampleOff;

                            const float inputVal = pInputRow[sX];
                            pOutputRow[x] += sampleWeight * inputVal;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int SparseFilterConvo::UpdateGradInput(lua_State *L)
{
    auto input = (const THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    auto gradOutput = (const THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    
    /*const int64_t kW = luaT_getfieldcheckint(L, 1, "m_kW");
    const int64_t kH = luaT_getfieldcheckint(L, 1, "m_kH");
    const int64_t dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    const int64_t dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    const int64_t kSize = kW * kH;*/

    const auto weights = get_mem_tensor<float>(L, "weight");
    //const auto bias = get_mem_tensor<float>(L, "bias");
    auto gradInput = get_mem_tensor<float>(L, "gradInput"); 
    //auto opProcMat = get_mem_tensor<float>(L, "opProcMat");
    auto sampleOffsets = get_mem_tensor<int>(L, "sampleOffsets");


    //cout << "Update Grad Input:" << endl
    //     << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
    //     << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

    THFloatTensor_zero(gradInput);

    //assert(THFloatTensor_nDimension(opProcMat) == 2);

    //cout << "OP Proc Mat: ["
    //     << THFloatTensor_size(opProcMat, 0) << " x "
    //     << THFloatTensor_size(opProcMat, 1) << "]"
    //     << endl;

    luaL_argcheck(L, 
                  input->nDimension == 4 && gradOutput->nDimension == 4, 
                  2, "Only a 4D (batch mode) tensor is supported");

    const int64_t nOutputPlane = luaT_getfieldcheckint(L, 1, "m_nOutputPlanes");
    const int64_t nInputPlane = luaT_getfieldcheckint(L, 1, "m_nInputPlanes");
    const int64_t nSamples = luaT_getfieldcheckint(L, 1, "m_numSamples");

    //cout << "Input Planes: " << nInputPlane << endl
    //     << "Output Planes: " << nOutputPlane << endl;

    const int64_t width = input->size[3];
    const int64_t height = input->size[2];

    //cout << "Width: " << width << endl
    //     << "Height: " << height << endl;

    const int64_t batchSize = input->size[0];
    const int64_t chanSize = width * height;
    const int64_t inputImgSize = nInputPlane * chanSize;
    const int64_t outputImgSize = nOutputPlane * chanSize;

    //cout << "Batch Size: " << batchSize << endl
    //     << "Channel Size: " << chanSize << endl
    //     << "Input Image Size: " << inputImgSize << endl
    //     << "Output Image Size: " << outputImgSize << endl;

    //const int64_t hkW = (kW / 2) * dkW;
    //const int64_t hkH = (kH / 2) * dkH;
    
    //cout << "Half Kernel Size: ["
    //     << hkW << ", " << hkH << "]" << endl;
    
    const float *pGradOutputData = THFloatTensor_data(gradOutput);
    float *pGradInputData = THFloatTensor_data(gradInput);
    const float *pWeightData = THFloatTensor_data(weights);
    const int *pOffsetData = THIntTensor_data(sampleOffsets);

    // Zero it out so that we can accumulate into the buffer
    THFloatTensor_zero(gradInput);

    #pragma omp parallel for
    for (int64_t imageIdx = 0; imageIdx < batchSize; ++imageIdx)
    {
        for (int64_t outputIdx = 0; outputIdx < nOutputPlane; ++outputIdx)
        {
            for (int64_t sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
            {
                const int64_t ySampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2];
                const int64_t xSampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2 + 1];

                const int64_t yStart = max<int64_t>(0, -ySampleOff);
                const int64_t yEnd = min(height, height - ySampleOff);

                const int64_t xStart = max<int64_t>(0, -xSampleOff);
                const int64_t xEnd = min(width, width - xSampleOff);

                for (int64_t inputIdx = 0; inputIdx < nInputPlane; ++inputIdx)
                {
                    const float sampleWeight = pWeightData[outputIdx * nInputPlane * nSamples + inputIdx * nSamples + sampleIdx];

                    float *pGradInputPlane = pGradInputData + imageIdx * inputImgSize + inputIdx * chanSize;
                    const float *pGradOutputPlane = pGradOutputData + imageIdx * outputImgSize + outputIdx * chanSize;

                    for (int64_t y = yStart; y < yEnd; ++y)
                    {
                        const int64_t sY = y + ySampleOff;

                        float *pGradInputRow = pGradInputPlane + sY * width;
                        const float *pGradOutputRow = pGradOutputPlane + y * width;

                        for (int64_t x = xStart; x < xEnd; ++x)
                        {
                            const int64_t sX = x + xSampleOff;

                            const float gradOutputVal = pGradOutputRow[x];
                            pGradInputRow[sX] += sampleWeight * gradOutputVal;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int SparseFilterConvo::AccGradParameters(lua_State *L)
{
    auto input = (const THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    auto gradOutput = (const THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    float scale = luaL_optnumber(L, 4, 1.f);

    const int64_t kW = luaT_getfieldcheckint(L, 1, "m_kW");
    const int64_t kH = luaT_getfieldcheckint(L, 1, "m_kH");
    const int64_t dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    const int64_t dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    
    const auto gradWeight = get_mem_tensor<float>(L, "gradWeight");
    const auto gradBias   = get_mem_tensor<float>(L, "gradBias");
    const auto gradBias2  = get_mem_tensor<float>(L, "gradBias2");
    auto opProcMat = get_mem_tensor<float>(L, "opProcMat");

    //cout << "Update Output:" << endl
    //     << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
    //     << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

    assert(THFloatTensor_nDimension(opProcMat) == 2);

    //cout << "OP Proc Mat: ["
    //     << THFloatTensor_size(opProcMat, 0) << " x "
    //     << THFloatTensor_size(opProcMat, 1) << "]"
    //     << endl;

    luaL_argcheck(L, input->nDimension == 4, 2, "Only a 4D (batch mode) tensor is supported");
   
    const int64_t nOutputPlane = luaT_getfieldcheckint(L, 1, "m_nOutputPlanes");
    const int64_t nInputPlane = luaT_getfieldcheckint(L, 1, "m_nInputPlanes");

    //cout << "Input Planes: " << nInputPlane << endl
    //     << "Output Planes: " << nOutputPlane << endl;

    const int64_t width = input->size[3];
    const int64_t height = input->size[2];

    //cout << "Width: " << width << endl
    //     << "Height: " << height << endl;

    // NOTE: Padding is assumed, so input sizes equal output sizes,
    // except for the number of planes
    const int64_t batchSize = input->size[0];
    const int64_t chanSize = width * height;
    const int64_t inputImgSize = nInputPlane * chanSize;

    const int64_t hkW = (kW / 2) * dkW;
    const int64_t hkH = (kH / 2) * dkH;
    
    auto govTensor = THFloatTensor_new();
    auto govSize = THLongStorage_newWithSize2(nOutputPlane, chanSize);
    auto wvTensor = THFloatTensor_new();
    auto transProc = THFloatTensor_new();
   
    const auto pInputData = THFloatTensor_data(input);
    auto pProcData = THFloatTensor_data(opProcMat);
    
    for (int64_t i = 0; i < batchSize; ++i)
    {
        //cout << "Processing Image: " << i << endl;

        const auto pInputImg = pInputData + i * inputImgSize;
        //auto pOutputImg = pOutputData + i * outputImgSize;

        //cout << "Creating output view" << endl;

        THFloatTensor_select(govTensor, (THFloatTensor*)gradOutput, 0, i);
        THFloatTensor_reshape(govTensor, govTensor, govSize); 

        // Sum into the bias
        THFloatTensor_sum(gradBias2, govTensor, 1);
        THFloatTensor_cadd(gradBias, gradBias, 1, gradBias2);

        for (int64_t k = 0; k < nInputPlane; ++k)
        {
            //cout << "Processing input plane: " << k << endl;

            // Get a matrix view of the weights for the current
            // input channel
            THFloatTensor_select(wvTensor, gradWeight, 0, k);

            assert(THFloatTensor_nDimension(wvTensor) == 2);

            const auto pInputChan = pInputImg + k * chanSize;

            int64_t r, c, mR = 0;
            for (int64_t kR = -hkH; kR <= hkH; kR += dkH)
            {
                for (int64_t kC = -hkW; kC <= hkW; kC += dkW, ++mR)
                {
                    //#pragma omp parallel for private(r)
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

                            pProcData[mR * chanSize + mC] = val;
                        }
                    }
                }
            }

            // gradOutput x input'
            THFloatTensor_transpose(transProc, opProcMat, 0, 1);

            THFloatTensor_addmm(wvTensor, 1.0f, wvTensor, scale, govTensor, transProc);
        }
 
    } 
   
    THFloatTensor_free(transProc);
    THFloatTensor_free(wvTensor);
    THLongStorage_free(govSize);
    THFloatTensor_free(govTensor);
    
    return 1;
}


} }
















