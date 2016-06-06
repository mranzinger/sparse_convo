
function init_full(a_layer, inputChans, h, w, stride)

    local t = torch.FloatTensor(inputChans, h, w)
    t:zero()

    for r = 1,h,stride do
        for c = 1,w,stride do
            t[{{},r,c}] = 1
        end
    end
    
    print(t:size())
    print(a_layer.weight:size())
    --t:reshape(h * w)

    for i=1,a_layer.weight:size(1) do
        a_layer.weight:select(1, i):cmul(t)
    end

    a_layer.bias:zero()

    return a_layer
end


