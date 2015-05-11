
function string.endsWith(str, strEnd)

    if string.len(strEnd) > string.len(str) then
        return false
    end

    return strEnd == '' or string.sub(str, -string.len(strEnd)) == strEnd

end

function string.endsWithAny(str, ...)

    for i=1, select('#', ...) do
        if string.endsWith(str, select(i, ...)) then
            return true
        end
    end

    return false

end

function lazy_ext(str)

    return string.sub(str, -4)

end

function table.sz(t)

    local ct = 0

    for k,v in pairs(t) do
        ct = ct + 1
    end

    return ct

end

function get_heat_num_correct(a_preds, a_labels, a_tolerance)

    assert(a_preds, 'Predictions tensor was not valid')
    assert(a_labels, 'Labels tensor was not valid')
    assert(a_preds:dim() == a_labels:dim(),
           'Preds and Labels tensors are not the same dimension')

    local subs = (a_labels - a_preds):abs()

    local tolerance = a_tolerance or 0.3

    local close = torch.le(subs, tolerance)

    return close:sum()

end

function get_heat_accuracy(a_preds, a_labels, a_tolerance)

    local ct = get_heat_num_correct(a_preds, a_labels)

    return ct / a_labels:nElement()

end

function sanitize_model(a_model)

    a_model:for_each(
        function(val)
            for name,field in pairs(val) do
                if torch.type(field) == 'cdata' then val[name] = nil end
                if name == 'homeGradBuffers' then val[name] = nil end
                if name == 'input_gpu' then val[name] = { } end
                if name == 'gradOutput_gpu' then val[name] = { } end
                if name == 'gradInput_gpu' then val[name] = { } end
                if (name == 'output' or name == 'gradInput') and
                    torch.type(field) == 'torch.CudaTensor' then
                    cutorch.withDevice(field:getDevice(),
                        function()
                            val[name] = field.new()
                        end
                    )
                end
            end
        end
    )

end












