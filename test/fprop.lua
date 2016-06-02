require 'sparseconvo'

v = torch.FloatTensor{

    { {
        {  1,  2,  3,  4,  5 },
        {  6,  7,  8,  9, 10 },
        { 11, 12, 13, 14, 15 },
        { 16, 17, 18, 19, 20 },
        { 21, 22, 23, 24, 25 }
    } },
    { {
        { 1, 1, 1, 1, 1 },
        { 1, 2, 2, 2, 1 },
        { 1, 2, 3, 2, 1 },
        { 1, 2, 2, 2, 1 },
        { 1, 1, 1, 1, 1 }
    } }
}

--v = v:view(2, 2, 4, 4)

--fsc = nn.SparseFilterConvo(2, 3, 3, 1, 1)
fsc = nn.SparseFilterConvo({ { 3, 2 } }, 9)
dc = nn.SpatialConvolutionMM(1, 2, 3, 3, 1, 1, 1, 1):float()

fsc:prepareSystem(v)

local i = 1
for k=1,2 do
    for y=-1, 1 do
        for x=-1, 1 do
            fsc.sampleOffsets[k][i][1] = y
            fsc.sampleOffsets[k][i][2] = x
        end
    end
end

fsc.weight:select(2, 1):fill(1)
fsc.weight:select(2, 2):fill(-1)

dc.weight:select(1, 1):fill(1)
dc.weight:select(1, 2):fill(-1)
dc.bias:zero()

print('Input:')
print(v)

print('Weights:')
print(fsc.weight)

op = fsc:forward(v)

top = dc:forward(v)

print('Output:')
print(op)

print('True Output:')
print(top)

assert(torch.all(torch.eq(op, top)), 'The output tensors were not the same')

print('TODO: Implement the backward convolutions')
os.exit(0)

fsc:zeroGradParameters()

-- Just use op as the gradient output
gi = fsc:backward(v, op)

--print('Gradient Input:')
--print(gi)

--dc = nn.SpatialConvolutionMM(1, 2, 5, 5, 1, 1, 2, 2):float()
--
--dcw = torch.FloatTensor
--{
--    1, 0, 1, 0, 1,
--    0, 0, 0, 0, 0,
--    1, 0, 1, 0, 1,
--    0, 0, 0, 0, 0,
--    1, 0, 1, 0, 1
--}
--
--dc.weight:select(1, 1):copy(dcw)
--dc.weight:select(1, 2):copy(dcw):mul(-1)





print('Gradient Input:')
print(gi)

dc:zeroGradParameters()
tgi = dc:backward(v, top)

print('True Gradient Input:')
print(tgi)

assert(torch.all(torch.eq(gi, tgi)), 'The gradient input tensors were not the same')

print('Grad Weight')
print(fsc.gradWeight)

print('True Grad Weight')
print(dc.gradWeight)

assert(torch.all(torch.eq(fsc.gradWeight, dc.gradWeight)))

print('Grad Bias')
print(fsc.gradBias)

print('True Grad Bias')
print(dc.gradBias)

assert(torch.all(torch.eq(fsc.gradBias, dc.gradBias)))

