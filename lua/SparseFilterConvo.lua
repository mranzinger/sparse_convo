require 'cutorch'
local mk = require 'multikey'

local SparseFilterConvo, parent = torch.class('nn.SparseFilterConvo',
                                              'nn.Module')

local roundf = function(x)
    if x < 0 then
        return math.ceil(x - 0.5)
    else
        return math.floor(x + 0.5)
    end
end

function SparseFilterConvo:__init(scales, numSamples)
    parent.__init(self)

    --Assumes stride of 1 and sufficient padding to preserve input
    --width and height

    local nOutputPlanes = 0
    for _, scale in ipairs(scales) do
        local targetSize, numFilters = unpack(scale)

        nOutputPlanes = nOutputPlanes + numFilters
    end

    self.m_nOutputPlanes = nOutputPlanes
    self.m_numSamples = numSamples

    self.sampleOffsets = torch.IntTensor(nOutputPlanes, numSamples, 2)

    local cOff = 0
    for _, scale in ipairs(scales) do
        local targetSize, numFilters = unpack(scale)
        assert(numSamples <= targetSize * targetSize, 'Invalid number of samples. Must be less than or equal to the target size squared')
        targetSize = (targetSize - 1) / 2

        for i=1, numFilters do
            local fTable = { }
            local fList = { }
            while #fList < numSamples do
                -- The result is that 50% of the samples will be within
                -- the target size. This is a reasonable sparsity goal
                local x = roundf(torch.normal(0, 1.349 * targetSize))
                local y = roundf(torch.normal(0, 1.349 * targetSize))

                if not mk.get(fTable, x, y) then
                    mk.put(fTable, x, y, true)
                    table.insert(fList, { y, x })
                end
            end

            -- Sort the samples by Y and then by X
            table.sort(fList,
                function(p1, p2)
                    if p1[1] < p2[1] then
                        return true
                    elseif p2[1] < p1[1] then
                        return false
                    end

                    return p1[2] < p2[2]
                end
            )

            local ct = self.sampleOffsets[cOff + i]
            for i, pt in ipairs(fList) do
                ct[i][1] = pt[1]
                ct[i][2] = pt[2]
            end
        end

        cOff = cOff + numFilters
    end
end

function SparseFilterConvo.gridInit(nOutputPlanes, kH, kW, sH, sW)

    local numSamples = kH * kW
    local ret = nn.SparseFilterConvo({ { numSamples, nOutputPlanes } }, numSamples)

    local hkH = (kH - 1) / 2
    local hkW = (kW - 1) / 2

    for k=1, nOutputPlanes do
        local i = 1
        for y=-hkH, hkH do
            for x=-hkW, hkW do
                ret.sampleOffsets[k][i][1] = y * sH
                ret.sampleOffsets[k][i][2] = x * sW
                i = i + 1
            end
        end
    end

    return ret

end

function SparseFilterConvo:reset(sdv)
    
    if not self.m_isPrepared then
        return
    end
    
    local weightSizes = torch.LongStorage{self.m_nOutputPlanes,
                                          self.m_nInputPlanes,
                                          self.m_numSamples}

    -- Use the Microsoft initialization method
    self.weight = torch.randn(weightSizes):float()
    self.gradWeight = torch.zeros(weightSizes):float()
    self.bias = torch.zeros(self.m_nOutputPlanes):float()
    self.gradBias = torch.zeros(self.m_nOutputPlanes):float()
    self.output = torch.FloatTensor()
    self.gradInput = torch.FloatTensor()

    local sdv = sdv or math.sqrt(2.0 / (self.m_nInputPlanes * self.m_numSamples))

    self.weight:mul(sdv)

    -- If CUDA, then move the weights to the device
    if self.m_isCuda then
        self.weight = self.weight:cuda()
        self.gradWeight = self.gradWeight:cuda()
        self.bias = self.bias:cuda()
        self.gradBias = self.gradBias:cuda()
        self.output = self.output:cuda()
        self.gradInput = self.gradInput:cuda()
        self.sampleOffsets = self.sampleOffsets:cuda()
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










