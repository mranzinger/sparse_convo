require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')


require 'build_network.lua'
require 'optim'

local net, crit = create_loc_net()
