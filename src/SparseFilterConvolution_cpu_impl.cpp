#include <iostream>
#include <assert.h>

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

    //const int64_t kW = luaT_getfieldcheckint(L, 1, "m_kW");
    //const int64_t kH = luaT_getfieldcheckint(L, 1, "m_kH");
    //const int64_t dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    //const int64_t dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    
    // Filter x Input Plane x Num Samples
    const auto weights = get_mem_tensor<float>(L, "weight");
    const auto bias   = get_mem_tensor<float>(L, "bias");
    //auto opProcMat = get_mem_tensor<float>(L, "opProcMat");
    auto output = get_mem_tensor<float>(L, "output");
    auto sampleOffsets = get_mem_tensor<int>(L, "sampleOffsets");

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

    auto pOutputData = THFloatTensor_data(output);
    const auto pInputData = THFloatTensor_data(input);
    const auto pWeightData = THFloatTensor_data(weights);
    const auto pBiasData = THFloatTensor_data(bias);
    const auto pOffsetData = THIntTensor_data(sampleOffsets);
    //auto pProcData = THFloatTensor_data(opProcMat);

    int64_t i,j,k;

    //cout << "Filling Biases" << endl;

    // Initialize the output to the biases
    for (i = 0; i < batchSize; ++i)
    {
        //#pragma omp parallel for private(j)
        for (j = 0; j < nOutputPlane; ++j)
        {
            auto pOutput = pOutputData + i * outputImgSize + j * chanSize;
            for (k = 0; k < chanSize; ++k)
            {
                pOutput[k] = pBiasData[j];
            }
        }
    }

    for (int64_t outputIdx = 0; outputIdx < nOutputPlane; ++outputIdx)
    {
        for (int64_t sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
        {
            const int32_t ySampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2];
            const int32_t xSampleOff = pOffsetData[outputIdx * nSamples * 2 + sampleIdx * 2 + 1];

            // We can compute the ranges where this specific part of the filter is valid
            const int32_t yStart = max(0, -ySampleOff);
            const int32_t yEnd = min(height, height - ySampleOff);

            const int32_t xStart = max(0, -xSampleOff);
            const int32_t xEnd = min(width, width - xSampleOff);

            for (int64_t inPlaneIdx = 0; inPlaneIdx < nInputPlane; ++inPlaneIdx)
            {
                for (int64_t y = yStart; y < yEnd; ++y)
                {
                    int64_t sY = y + ySampleOff;

                    for (int64_t x = xStart; x < xEnd; ++x)
                    {
                        int64_t sX = x + xSampleOff;

                        for (int64_t inputIdx = 0; inputIdx < nInputPlane; ++inputIdx)
                        {
                            const float sampleWeight = pWeightData[outputIdx * nInputPlane * nSamples + inputIdx * nSamples + sampleIdx];

                            for (int64_t imageIdx = 0; imageIdx < batchSize; ++imageIdx)
                            {
                                const float inputVal = pInputData[imageIdx * inputImgSize + inputIdx * chanSize + sY * width + sX];

                                pOutputData[imageIdx * outputImgSize + outputIdx * chanSize + y * width + x] += sampleWeight * inputVal;
                            }
                        }                
                    }
                }
            }
        }
    }

    //const int64_t hkW = (kW / 2) * dkW;
    //const int64_t hkH = (kH / 2) * dkH;

    //cout << "Half Kernel Size: ["
    //     << hkW << ", " << hkH << "]" << endl;

    /*auto wvTensor = THFloatTensor_new();
    auto ovTensor = THFloatTensor_new();
    auto ovSize = THLongStorage_newWithSize2(nOutputPlane, chanSize);

    //cout << "Created temporary tensors" << endl;

    for (i = 0; i < batchSize; ++i)
    {
        //cout << "Processing Image: " << i << endl;

        const auto pInputImg = pInputData + i * inputImgSize;
        //auto pOutputImg = pOutputData + i * outputImgSize;

        //cout << "Creating output view" << endl;

        // Create a matrix view of the current output buffer
        THFloatTensor_select(ovTensor, output, 0, i);

        //cout << "Selected current output image" << endl;

        THFloatTensor_reshape(ovTensor, ovTensor, ovSize);

        assert(THFloatTensor_nDimension(ovTensor) == 2);

        //cout << "Output view: ["
        //     << THFloatTensor_size(ovTensor, 0) << " x "
        //     << THFloatTensor_size(ovTensor, 1) << "]"
        //     << endl;

        for (k = 0; k < nInputPlane; ++k)
        {
            //cout << "Processing input plane: " << k << endl;

            // Get a matrix view of the weights for the current
            // input channel
            THFloatTensor_select(wvTensor, weights, 0, k);

            assert(THFloatTensor_nDimension(wvTensor) == 2);


            //cout << "Weight view: ["
            //     << THFloatTensor_size(wvTensor, 0) << " x "
            //     << THFloatTensor_size(wvTensor, 1) << "]"
            //     << endl;

            const auto pInputChan = pInputImg + k * chanSize;

            int64_t r, c, mR = 0;
            //for (int64_t kR = -hkH; kR <= hkH; kR += dkH)
            //{
                //for (int64_t kC = -hkW; kC <= hkW; kC += dkW, ++mR)
                //{
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
                //}
            //}

            //Now that the matrix is filled out, do the multiply and accumulate into
            // the ovTensor view
            THFloatTensor_addmm(ovTensor, 1.0f, ovTensor, 1.0f, wvTensor, opProcMat);
        }
    }

    // Free the processing buffers
    THFloatTensor_free(wvTensor);
    THFloatTensor_free(ovTensor);
    THLongStorage_free(ovSize);*/

    return 0;
}

int SparseFilterConvo::UpdateGradInput(lua_State *L)
{
    auto input = (const THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    auto gradOutput = (const THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
    
    const int64_t kW = luaT_getfieldcheckint(L, 1, "m_kW");
    const int64_t kH = luaT_getfieldcheckint(L, 1, "m_kH");
    const int64_t dkW = luaT_getfieldcheckint(L, 1, "m_dkW");
    const int64_t dkH = luaT_getfieldcheckint(L, 1, "m_dkH");
    const int64_t kSize = kW * kH;

    const auto weights = get_mem_tensor<float>(L, "weight");
    const auto bias = get_mem_tensor<float>(L, "bias");
    auto gradInput = get_mem_tensor<float>(L, "gradInput"); 
    auto opProcMat = get_mem_tensor<float>(L, "opProcMat");

    //cout << "Update Grad Input:" << endl
    //     << "\tKernel Size: [" << kW << " x " << kH << "]" << endl
    //     << "\tKernel Stride: [" << dkW << " x " << dkH << "]" << endl;

    THFloatTensor_zero(gradInput);

    assert(THFloatTensor_nDimension(opProcMat) == 2);

    //cout << "OP Proc Mat: ["
    //     << THFloatTensor_size(opProcMat, 0) << " x "
    //     << THFloatTensor_size(opProcMat, 1) << "]"
    //     << endl;

    luaL_argcheck(L, 
                  input->nDimension == 4 && gradOutput->nDimension == 4, 
                  2, "Only a 4D (batch mode) tensor is supported");

    const int64_t nOutputPlane = luaT_getfieldcheckint(L, 1, "m_nOutputPlanes");
    const int64_t nInputPlane = luaT_getfieldcheckint(L, 1, "m_nInputPlanes");

    //cout << "Input Planes: " << nInputPlane << endl
    //     << "Output Planes: " << nOutputPlane << endl;

    const int64_t width = input->size[3];
    const int64_t height = input->size[2];

    //cout << "Width: " << width << endl
    //     << "Height: " << height << endl;

    const int64_t batchSize = input->size[0];
    const int64_t chanSize = width * height;
    //const int64_t inputImgSize = nInputPlane * chanSize;
    //const int64_t outputImgSize = nOutputPlane * chanSize;

    //cout << "Batch Size: " << batchSize << endl
    //     << "Channel Size: " << chanSize << endl
    //     << "Input Image Size: " << inputImgSize << endl
    //     << "Output Image Size: " << outputImgSize << endl;

    const int64_t hkW = (kW / 2) * dkW;
    const int64_t hkH = (kH / 2) * dkH;
    
    //cout << "Half Kernel Size: ["
    //     << hkW << ", " << hkH << "]" << endl;
         
    auto wvTensor = THFloatTensor_new();
    auto givTensor = THFloatTensor_new();
    auto govTensor = THFloatTensor_new();
    auto govSize = THLongStorage_newWithSize2(nOutputPlane, chanSize);
    auto procVTensor = THFloatTensor_new();

    //cout << "Created temporary tensors" << endl;
    
    int64_t i, k;
    for (i = 0; i < batchSize; ++i)
    {
        //cout << "Processing Image: " << i << endl;

        //cout << "Creating output grad view" << endl;

        THFloatTensor_select(govTensor, (THFloatTensor*)gradOutput, 0, i);

        //cout << "Selected current grad output image" << endl;

        THFloatTensor_reshape(govTensor, govTensor, govSize);

        assert(THFloatTensor_nDimension(govTensor) == 2);

        for (k = 0; k < nInputPlane; ++k)
        {
            // Select part of the proc mat for the current input channel
            THFloatTensor_narrow(procVTensor, opProcMat, 0, kSize * k, kSize);

            // Select the current set of weights
            THFloatTensor_select(wvTensor, weights, 0, k);

            assert(THFloatTensor_nDimension(wvTensor) == 2);

            // Transpose the weights such that they are:
            // kSize x nOutputPlane
            THFloatTensor_transpose(wvTensor, wvTensor, 0, 1);

            assert(THFloatTensor_size(wvTensor, 0) == kSize);
            assert(THFloatTensor_size(wvTensor, 1) == nOutputPlane);

            THFloatTensor_addmm(procVTensor, 0.0f, procVTensor, 1.0f, wvTensor, govTensor);
        }
        
        THFloatTensor_select(givTensor, gradInput, 0, i);

        auto giData = THFloatTensor_data(givTensor);
        const auto unrollData = THFloatTensor_data(opProcMat);

        // Ok, so now the proc mat has been filled, so we simply need to copy the respective
        // elements back into the input grad buffer
        int64_t iChan = 0;
        //#pragma omp parallel for private(iChan)
        for (iChan = 0; iChan < nInputPlane; ++iChan)
        {
            auto giChanData = giData + iChan * chanSize;

            k = 0;
            for (int64_t kR = -hkH; kR <= hkH; kR += dkH)
            {
                for (int64_t kC = -hkW; kC <= hkW; kC += dkW, ++k)
                {
                    auto rowUnroll = unrollData + iChan * kSize * chanSize + k * chanSize;

                    for (int64_t r = 0; r < height; ++r)
                    {
                        int64_t ipR = r + kR;

                        if (ipR >= 0 && ipR < height)
                        {
                            for (int64_t c = 0; c < width; ++c)
                            {
                                int64_t ipC = c + kC;

                                if (ipC >= 0 && ipC < width)
                                {
                                    const float val = rowUnroll[r * width + c];       

                                    giChanData[ipR * width + ipC] += val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Free the processing tensors
    THFloatTensor_free(wvTensor);
    THFloatTensor_free(givTensor);
    THFloatTensor_free(govTensor);
    THFloatTensor_free(procVTensor);
    THLongStorage_free(govSize);

    return 1;
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
















