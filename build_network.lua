package.path = '/home/mranzinger/dev/torch-transition/lua/?.lua;' .. package.path

require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'fbcunn'
require 'fbnn'

require 'MySigmoid'
require 'MyBCECriterion'

function create_loc_net()

    local model = nn.Sequential()

    create_33_module(model, 3, 32, 4)

    model:add(cudnn.SpatialMaxPooling(2, 2,
                                      2, 2))

    create_33_module(model, 32, 64, 4) 

    model:add(cudnn.SpatialMaxPooling(2, 2,
                                      2, 2))

    create_33_module(model, 64, 128, 3)

    model:add(cudnn.SpatialMaxPooling(2, 2,
                                      2, 2))

    create_33_module(model, 128, 64, 1)

    --model:add(nn.SpatialZeroPadding(10, 10, 10, 10))

    --model:add(nn.SpatialConvolutionCuFFT(64,
    --                                     32,
    --                                     21, 21))

    create_nn_module(model, 64, 64, 5, 7)

    --create_nn_module(model, 32, 32, 1, 13)
    --create_33_module(model, 64, 32, 2)

    model:add(cudnn.SpatialConvolution(64,
                                       1,
                                       1, 1,
                                       1, 1,
                                       0, 0))

    model:add(nn.MySigmoid(true))

    for i=1,model:size() do
        init_layer_ms(model:get(i))
    end

    crit = nn.MyBCECriterion()
    crit.fullAverage = true
        
    return { err = 100000, model = model:cuda() }, crit:cuda()

end

function create_33_module(a_model, a_from, a_to, a_num)

    create_nn_module(a_model, a_from, a_to, a_num, 3)

end

function create_nn_module(a_model, a_from, a_to, a_num, a_k)
    
    local pad = (a_k - 1) / 2

    print(pad)

    local prev = a_from

    for i=1,a_num do
        a_model:add(cudnn.SpatialConvolution(prev,
                                             a_to,
                                             a_k, a_k,
                                             1, 1,
                                             pad, pad))
        a_model:add(cudnn.ReLU(true))

        prev = a_to
    end

end

function init_layer_ms(a_layer)
    --a_layer:reset()
   
    -- Implement the initialization
    -- described in the Microsoft paper:
    -- http://arxiv.org/pdf/1502.01852v1.pdf 
    local dw = a_layer.weight
    local lb = a_layer.bias

    if dw then

        local fanIn = dw[1]:nElement()

        print('Fan In:', fanIn)

        local stdVal = torch.sqrt(2.0 / fanIn)

        print('Std Val:', stdVal)

        -- Create a host tensor filled with random
        -- values. Mean: 0, Variance (& std): 1
        local hw = torch.randn(dw:size())

        -- Multiply the values by the stdVal to change the
        -- standard deviation to the target value
        hw:mul(stdVal)

        dw:copy(hw)
    
    end
    
    if lb then
        
        lb:zero()
    
    end
    
    return a_layer
end


