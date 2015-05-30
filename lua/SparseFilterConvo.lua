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
                                          self.m_kH * self.m_kW}

    -- Use the Microsoft initialization method
    self.weight = torch.randn(weightSizes):float()
    self.gradWeight = torch.FloatTensor()
    self.bias = torch.zeros(self.m_nOutputPlanes):float()
    self.gradBias = torch.FloatTensor()
    self.output = torch.FloatTensor()
    self.opProcMat = torch.FloatTensor()
    self.gradInput = torch.FloatTensor()

    local sdv = sdv or math.sqrt(2.0 / (self.m_nInputPlanes * self.m_kH * self.m_kW))

    self.weight:mul(sdv)

    -- If CUDA, then move the weights to the device
    if self.m_isCuda then
        self.weight = self.weight:cuda()
        self.gradWeight = self.gradWeight:cuda()
        self.bias = self.bias:cuda()
        self.gradBias = self.gradBias:cuda()
        self.output = self.output:cuda()
        self.opProcMat = self.opProcMat:cuda()
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

    -- op += w_i * P
    -- w_i: input channel weights
    -- P: unrolled input matrix
    --      each row is the kernel about x,y
    --      each column is an x,y pixel
    self.opProcMat:resize(self.m_kW * self.m_kH, vInput:size(3) * vInput:size(4))

    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_updateOutput(self, vInput)
    else
        input.nn.SparseFilterConvo_cpu_updateOutput(self, vInput)
    end

    return self.output
     
end

function SparseFilterConvo:updateGradInput(input, gradOutput)
    
    assert(self.m_isCuda == self:isCuda(input))
    assert(self.m_isCuda == self:isCuda(gradOutput))
    
    local vInput
    if input:dim() == 3 then
        vInput = input:view(1, input:size(1), input:size(2), input:size(3))
    elseif input:dim() == 4 then
        vInput = input
    else
        error('Invalid input dimension!')
    end

    local vGradOutput
    if gradOutput:dim() == 3 then
        vGradOutput = gradOutput:view(1, gradOutput:size(1), gradOutput:size(2), gradOutput:size(3))
    elseif gradOutput:dim() == 4 then
        vGradOutput = gradOutput
    else
        error('Invalid gradient output dimension!')
    end

    self.gradInput:resize(input:size())

    self.opProcMat:resize(self.m_nInputPlanes * self.m_kW * self.m_kH,
                          vInput:size(3) * vInput:size(4))

    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_updateGradInput(self, vInput, vGradOutput)
    else
        input.nn.SparseFilterConvo_cpu_updateGradInput(self, vInput, vGradOutput)
    end

    return self.gradInput
end

function SparseFilterConvo:accGradParameters(input, gradOutput, scale)

    assert(self.m_isCuda == self:isCuda(input))
    assert(self.m_isCuda == self:isCuda(gradOutput))

    local scale = scale or 1
   
    local vInput
    if input:dim() == 3 then
        vInput = input:view(1, input:size(1), input:size(2), input:size(3))
    elseif input:dim() == 4 then
        vInput = input
    else
        error('Invalid input dimension!')
    end

    local vGradOutput
    if gradOutput:dim() == 3 then
        vGradOutput = gradOutput:view(1, gradOutput:size(1), gradOutput:size(2), gradOutput:size(3))
    elseif gradOutput:dim() == 4 then
        vGradOutput = gradOutput
    else
        error('Invalid gradient output dimension!')
    end
 
    self.gradWeight:resize(self.weight:size())
    self.gradBias:resize(self.bias:size())

    self.opProcMat:resize(self.m_kW * self.m_kH, vInput:size(3) * vInput:size(4))
 
    if self.m_isCuda then
        input.nn.SparseFilterConvo_cu_accGradParameters(self, vInput, vGradOutput, scale)
    else
        input.nn.SparseFilterConvo_cpu_accGradParameters(self, vInput, vGradOutput, scale)
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

    -- Initialize the weights
    self:reset()

end










