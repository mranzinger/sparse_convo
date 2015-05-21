require 'cutorch'

local SparseFilterConvo, parent = torch.class('nn.SparseFilterConvo',
                                              'nn.Module')

function SparseFilterConvo:__init(nOutputPlanes,
                                  kW, kH,
                                  dkW, dkH)
    --Assumes stride of 1 and sufficient padding to preserve input
    --width and height
    self.m_nOutputPlanes = nOutputPlanes
    self.m_kW = kW
    self.m_kH = kH
    self.m_dkW = dkW or 2
    self.m_dkH = dkH or 2
end

function SparseFilterConvo:reset()
    
    -- Input planes are the outer dimension so that we can stream in
    -- planes and accumulate partial convolutions into the output buffer
    local weightSizes = torch.LongStorage(self.m_nInputPlanes,
                                          self.m_nOutputPlanes,
                                          self.m_kH,
                                          self.m_kW)

    -- Use the Microsoft initialization method
    self.weights = torch.randn(weightSizes):float()
    self.bias = torch.zeros(self.m_nOutputPlanes):float()

    local sdv = math.sqrt(2.0 / (self.m_nInputPlanes * self.m_kH * self.m_kW))

    self.weights:mul(sdv)

    -- If CUDA, then move the weights to the device
    if self.m_isCuda then
        self.weights = self.weights:cuda()
        self.bias = self.bias:cuda()
    end

end

function SparseFilterConvo:updateOutput(input)
    --TODO: Implement Me!
    
    if torch.type(input) == 'torch.FloatTensor' then 
        input.nn.SparseFilterConvo_cpu_updateOutput(self, input)
    elseif torch.type(input) == 'torch.CudaTensor' then
        input.nn.SparseFilterConvo_cu_updateOutput(self, input)
    else
        error('Invalid tensor type: ' .. torch.type(input))
    end

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

function SparseFilterConvo::isCuda(ts)

    if not ts then
        error('Invalid nil tensor')
    elseif torch.type(ts) == 'torch.CudaTensor' then
        return true
    elseif torch.type(ts) == 'torch.FloatTensor' then
        return false
    else
        error('Unsupported tensor type: ' .. torch.type(ts))
    end

end

function SparseFilterConvo::prepareSystem(ts)

    local isCuda = self:isCuda(ts)

    if self.m_isPrepared then
        assert(self.m_isCuda == isCuda)
        return
    end

    self.m_isCuda = isCuda

    local chanIdx = 1

    if ts:dim() == 4 then
        chanIdx = 2
    end

    self.m_nInputPlanes = ts:size(chanIdx)

    -- Initialize the weights
    self:reset()

end










