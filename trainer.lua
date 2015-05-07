require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')


require 'build_network.lua'
require 'optim'
require 'data_loader'

local net, crit = create_loc_net()

local loader = DataLoader()
loader:find_examples('/cuda-ssd/icdar/2015/neocr')

-- Let the loader study the network so that it knows how to prepare
-- the labels
loader:study_net(net.model)

loader:finalize()

d,l = loader:get_train_example()
