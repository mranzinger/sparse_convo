
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

function table.sz(t)

    local ct = 0

    for k,v in pairs(t) do
        ct = ct + 1
    end

    return ct

end
