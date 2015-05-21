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

function SparseFilterConvo:reset(sdv)
    
    if not self.m_isPrepared then
        return
    end
    
    -- Input planes are the outer dimension so that we can stream in
    -- planes and accumulate partial convolutions into the output buffer
    local weightSizes = torch.LongStorage{self.m_nInputPlanes,
                                          self.m_nOutputPlanes,
                                          self.m_kH,
                                          self.m_kW}

    -- Use the Microsoft initialization method
    self.weight = torch.randn(weightSizes):float()
    self.bias = torch.zeros(self.m_nOutputPlanes):float()
    self.output = torch.FloatTensor()
    self.gradInput = torch.FloatTensor()

    local sdv = sdv or math.sqrt(2.0 / (self.m_nInputPlanes * self.m_kH * self.m_kW))

    self.weight:mul(sdv)

    -- If CUDA, then move the weights to the device
    if self.m_isCuda then
        self.weight = self.weight:cuda()
        self.bias = self.bias:cuda()
        self.output = self.output:cuda()
        self.gradInput = self.gradInput:cuda()
    end

end

function SparseFilterConvo:updateOutput(input)
    
    self:prepareSystem(input)

    local vInput
    if input:dim() == 3 then
        vInput = input:view(1, input:size(1), input:size(2), input:size(3))
    elseif input:dim() == 4 then
        vInput = input
    else
        error('Invalid input dimension!')
    end

    self.output:resize(vInput:size(1),
                       self.m_nOutputPlanes,
                       vInput:size(3),
                       vInput:size(4))

    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_updateOutput(self, input)
    else
        input.nn.SparseFilterConvo_cpu_updateOutput(self, input)
    end

    return self.output
     
end

function SparseFilterConvo:updateGradInput(input, gradOutput)
    
    assert(self.m_isCuda == self:isCuda(input))
    assert(self.m_isCuda == self:isCuda(gradOutput))
    
    self.gradInput:resize(input:size())

    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_updateGradInput(self, input, gradOutput)
    else
        input.nn.SparseFilterConvo_cpu_updateGradInput(self, input, gradOutput)
    end

    return self.gradInput
end

function SparseFilterConvo:accGradParameters(input, gradOutput, scale)

    assert(self.m_isCuda == self:isCuda(input))
    assert(self.m_isCuda == self:isCuda(gradOutput))

    local scale = scale or 1
    
    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_accGradParameters(self, input, gradOutput, scale)
    else
        input.nn.SparseFilterConvo_cpu_accGradParameters(self, input, gradOutput, scale)
    end

end

function SparseFilterConvo:isCuda(ts)

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

function SparseFilterConvo:prepareSystem(ts)

    local isCuda = self:isCuda(ts)

    if self.m_isPrepared then
        assert(self.m_isCuda == isCuda)
        return
    end

    self.m_isPrepared = true
    self.m_isCuda = isCuda

    local chanIdx = 1

    if ts:dim() == 4 then
        chanIdx = 2
    end

    self.m_nInputPlanes = ts:size(chanIdx)

    print('# Input Planes:', self.m_nInputPlanes)

    -- Initialize the weights
    self:reset()

end










