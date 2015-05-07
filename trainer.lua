require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')


require 'build_network.lua'
require 'optim'
require 'data_loader'

local net, crit = create_loc_net()

local loader
if paths.filep('loader.t7') then

    loader = torch.load('loader.t7')

else

    loader = DataLoader()
    loader:find_examples('/cuda-ssd/icdar/2015/neocr')

    loader:finalize()

    torch.save('loader.t7', loader)

end

local model = net.model

-- Let the loader study the network so that it knows how to prepare
-- the labels
loader:study_net(model)

local optimState = {
    learningRate = 0.01,
    momentum     = 0.90,
    weightDecay  = 0.0005
}

local optimizer = nn.Optim(model, optimState)

model:cuda()

--d,l = loader:get_train_example()

print('Batch #', 'Loss', 'Accuracy', 'Total Time (s)')

gpuData = torch.CudaTensor()
gpuLabels = torch.CudaTensor()

timer = torch.Timer()

local batchCounter = 0

local bestErr = net.err

while true do

    print('Training!')
    model:training()
    for i=1,1000 do
        timer:reset()

        local data, labels = get_train_example()
        
        gpuData:resize(data:size()):copy(data)
        gpuLabels:resize(labels:size()):copy(labels)

        local err, pred = optimizer:optimize(
                                        optim.sgd,
                                        gpuData,
                                        gpuLabels,
                                        crit) 
        cutorch.synchronize()

        local accuracy = get_accuracy(pred:float(), labels)

        local totalTime = timer:time().real

        local s = string.format("%8.0f    %10.6f    %6.4f    %8.5fs",
                                batchCounter, err, accuracy, totalTime)
        
        print(s)
        
        batchCounter = batchCounter + 1          

        if i % 5 == 0 then
            collectgarbage()
        end

    end

    print('Testing!')
    model:evaluate()
    local totalCorrect = 0
    local totalErr = 0
    
    for i=1,loader:num_test_examples() do

        local data, labels = get_test_example(i)

        gpuData:resize(data:size()):copy(data)
        gpuLabels:resize(labels:size()):copy(labels)

        local output = model:forward(gpuData)
        local err = crit:forward(output, labels)
        cutorch.synchronize()
        local pred = output:float()

        totalCorrect = totalCorrect + get_num_correct(pred, labels)
        totalErr = totalErr + err

    end

    local accuracy = totalCorrect / loader:num_test_examples()
    local err = totalErr / loader:num_test_examples()

    local s = string.format("TEST    %10.6f    %6.4f",
                            err, accuracy)
    print(s)    

    net.err = err

    torch.save('last.t7', net)

    if err < bestErr then
        bestErr = err

        torch.save('best.t7', net)
    end


end








