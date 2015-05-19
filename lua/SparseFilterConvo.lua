require 'cutorch'

local SparseFilterConvo, parent = torch.class('nn.SparseFilterConvo',
                                              'nn.Module')

function SparseFilterConvo:__init()
    --TODO: Implement Me!
end

function SparseFilterConvo:reset(stdv)
    --TODO: Implement Me!
end

function SparseFilterConvo:updateOutput(input)
    --TODO: Implement Me!
    input.nn.SparseFilterConvo_updateOutput(self, input)

    return self.output
end

function SparseFilterConvo:updateGradInput(input, gradOutput)
    --TODO: Implement Me!
    input.nn.SparseFilterConvo_updateGradInput(self, input, gradOutput)

    return self.gradInput
end

function SparseFilterConvo:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    --TODO: Implement Me!
    input.nn.SparseFilterConvo_accGradParameters(self, input, gradOutput, scale)
end


