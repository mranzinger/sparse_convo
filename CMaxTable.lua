
local CMaxTable, parent = torch.class('nn.CMaxTable', 'nn.Module')

function CMaxTable:__init()
    parent.__init(self)

    self.gradInput = { }
    self.masks = { }
end

function CMaxTable:updateOutput(input)

    assert(torch.type(input) == 'table', 'Input must be a table')

    self.output:resizeAs(input[1]):copy(input[1])

    for i=2, #input do
        local mask = self.masks[i]

        if not mask then
            mask = torch.gt(input[i], self.output)
        else
            torch.gt(mask, input[i], self.output)
        end
        self.masks[i] = mask

        self.output:maskedCopy(mask, input[i])
    end

    return self.output

end

function CMaxTable:updateGradInput(input, gradOutput)

    for i=1, #input do
        self.gradInput[i] = self.gradInput[i] or input[i].new()

        self.gradInput[i]:resizeAs(input[i])
    end

    -- Zero out everywhere that was greater in one of the other buffers
    self.gradInput[1]:copy(gradOutput)
    for i=2, #input do
        self.gradInput[1]:maskedFill(self.masks[i], 0)
    end

    -- The rest aren't special snowflakes
    for i=2, #input do
        
        self.gradInput[i]:zero()
        -- We know that up to this point, these maxes are valid against
        -- all lesser indexes
        self.gradInput[i]:maskedCopy(self.masks[i], gradOutput)

        -- Now, we just need to zero out the bits set in greater indexes
        for k=i+1, #input do
            self.gradInput[k]:maskedFill(self.masks[k], 0)
        end
    end

    return self.gradInput

end
