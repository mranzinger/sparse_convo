require 'torch'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(3)

require 'build_network.lua'
require 'optim'
require 'data_loader'
require 'utils'
require 'train_utils'

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

if paths.filep('best.t7') then
    print('Found an existing checkpoint. Loading.')
    net = torch.load('best.t7')

    print('Error:', net.err, 'Accuracy:', net.acc)
end

local model = net.model

-- Let the loader study the network so that it knows how to prepare
-- the labels
loader:study_net(model)

local optimState = {
    learningRate = net.lr or 0.01,
    momentum     = 0.90,
    weightDecay  = 0.0005
}

print('Learning Rate:', optimState.learningRate)

local optimizer = nn.Optim(model, optimState)

model:cuda()

--d,l = loader:get_train_example()

local pwd = lfs.currentdir()

print('Current Dir:', pwd)

print('Batch #', 'Loss', 'Accuracy', 'Total Time (s)')

gpuData = torch.CudaTensor()
gpuLabels = torch.CudaTensor()

timer = torch.Timer()

local batchCounter = net.bc or 0

local bestErr = net.err

local nbCt = 0

while true do

    print('Training!')
    model:training()
    for i=1,100 do
        timer:reset()

        local data, labels = loader:get_train_example()
       
        if data:dim() == 3 then
            data = data:view(1, data:size(1), data:size(2), data:size(3))
        end
        
        if labels:dim() == 3 then
            labels = labels:view(1, labels:size(1), labels:size(2), labels:size(3))
        end 
        
        gpuData:resize(data:size()):copy(data)
        gpuLabels:resize(labels:size()):copy(labels)

        local err, pred = optimizer:optimize(
                                        optim.sgd,
                                        gpuData,
                                        gpuLabels,
                                        crit) 
        cutorch.synchronize()

        local accuracy = get_heat_accuracy(pred:float(), labels)

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
    local totalAcc = 0
    local totalErr = 0
    
    if not paths.dirp('preds_last') then
        lfs.mkdir('preds_last')
        lfs.mkdir('preds_last/image')
        lfs.mkdir('preds_last/gt')
        lfs.mkdir('preds_last/pred')
    end


    for i=1,loader:num_test_examples() do

        local data, labels = loader:get_test_example(i)

        gpuData:resize(data:size()):copy(data)
        gpuLabels:resize(labels:size()):copy(labels)

        local output = model:forward(gpuData)
        local err = crit:forward(output, gpuLabels)
        cutorch.synchronize()
        local pred = output:float()

        totalAcc = totalAcc + get_heat_accuracy(pred, labels)
        totalErr = totalErr + err

        local exImPath = loader.m_test[i].im
        local exGtPath = loader.m_test[i].gt

        lfs.chdir(pwd .. '/preds_last/image')
        os.execute('cp ' .. exImPath .. ' ' .. tostring(i) .. lazy_ext(exImPath)) 
        lfs.chdir(pwd .. '/preds_last/gt')
        image.save(tostring(i) .. '.png', labels)
        lfs.chdir(pwd .. '/preds_last/pred')
        image.save(tostring(i) .. '.png', pred)
        lfs.chdir(pwd)

    end

    local accuracy = totalAcc / loader:num_test_examples()
    local err = totalErr / loader:num_test_examples()

    local s = string.format("TEST    %10.6f    %6.4f",
                            err, accuracy)
    print(s)    

    net.err = err
    net.acc = accuracy
    net.bc = batchCounter
    net.lr = optimState.learningRate

    sanitize_model(model)
    torch.save('last.t7', net)

    if err < bestErr then
        bestErr = err
        nbCt = 0

        print('This checkpoint is the new best!!!')

        torch.save('best.t7', net)

        os.execute('rm -rf preds_best')
        os.execute('cp -r preds_last preds_best')
    else
        nbCt = nbCt + 1 

        if nbCt == 5 then
            optimState.learningRate = optimState.learningRate / 2
            print('NEW LEARNING RATE:', optimState.learningRate)
            nbCt = 0
        end
    end


end








