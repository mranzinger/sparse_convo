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

fsc = nn.SparseFilterConvo(2, 3, 3, 2, 2)

fsc:prepareSystem(v)

fsc.weight:select(2, 1):fill(1)
fsc.weight:select(2, 2):fill(-1)

print('Input:')
print(v)

print('Weights:')
print(fsc.weight)

op = fsc:forward(v)

print('Output:')
print(op)

-- Just use op as the gradient output
gi = fsc:backward(v, op)

print('Gradient Input:')
print(gi)

dc = nn.SpatialConvolutionMM(1, 2, 5, 5, 1, 1, 2, 2):float()

dcw = torch.FloatTensor
{
    1, 0, 1, 0, 1,
    0, 0, 0, 0, 0,
    1, 0, 1, 0, 1,
    0, 0, 0, 0, 0,
    1, 0, 1, 0, 1
}

dc.weight:select(1, 1):copy(dcw)
dc.weight:select(1, 2):copy(dcw):mul(-1)
dc.bias:zero()

top = dc:forward(v)

print('True Output:')
print(top)

tgi = dc:backward(v, top)

print('True Gradient Input:')
print(tgi)

assert(torch.all(torch.eq(op, top)), 'The output tensors were not the same')
assert(torch.all(torch.eq(gi, tgi)), 'The gradient input tensors were not the same')
