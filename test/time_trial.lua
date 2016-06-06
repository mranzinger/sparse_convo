require 'sparseconvo'
require 'test_util'

print("Starting Tests!")

local inputChans = 3

print("Creating large input")

input = torch.randn(128, inputChans, 224, 224):float()

print("Creating calc table")

calcTable = { }

for k = 3,9,2 do

    for s = 1, 5 do

        dk = (k - 1) * s + 1
        table.insert(calcTable, {
            tostring(k) .. "x" .. tostring(k) .. ", stride " .. tostring(s),
            nn.SparseFilterConvo.gridInit(32, k, k, s, s),
            init_full(nn.SpatialConvolutionMM(inputChans, 32, dk, dk, 1, 1, s, s):float(), inputChans, dk, dk, s)
        }) 

    end

end

print(calcTable)

print("Running tests!\n\n")

timer = torch.Timer()

names = { "Sparse:", "Dense:" }

for i=1,#calcTable do

    local entry = calcTable[i]

    print(entry[1])

    for n = 1, #names do
        print(names[n])

        -- Warm up the layer
        entry[n+1]:forward(input)

        timer:reset()

        for j = 1, 10 do
            op = entry[n+1]:forward(input)

            --entry[n+1]:zeroGradParameters()

            --gi = entry[n+1]:backward(input, op)
        end

        print(timer:time().real .. "s")

        collectgarbage()
    end

    print("\n\n")
end

