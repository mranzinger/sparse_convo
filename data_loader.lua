require 'torch'
require 'lfs'
require 'utils'

local data_loader = torch.class('DataLoader')

function data_loader:__init(a_dsDir)

    self:find_examples(a_dsDir)

    self:allocate_holdout()

end

function data_loader:find_examples(a_dsDir)

    assert(paths.dirp(a_dsDir), 'The dataset directory doesn\'t exist!')

    local locDir = a_dsDir .. '/loc'
    local segDir = a_dsDir .. '/seg'

    local w = { 'train', 'test' }

    local examples = { }

    for i=1,#w do

        local v = w[i]

        local locRoot = locDir .. '/' .. v
        local segRoot = segDir .. '/' .. v

        local y = { locRoot, segRoot }

        for k=1,#y do
            local z = y[k]

            assert(paths.dirp(z), 'Invalid dataset structure')

            for p in lfs.dir(z) do
                
                if string.endsWithAny(p, '.png', '.jpg') then
                
                    local isTruth = string.find(p, 'GT') ~= nil

                    local numS, numE = string.find(p, '%d+')

                    if not numS then
                        print('Invalid path:', p)
                    else

                        local num = p:sub(numS, numE)

                        --print(p,': Num:', num,': Is GT:', isTruth)

                        local key = v .. '_' .. num

                        local entry = examples[key]
                        if not entry then
                            entry = { }
                            examples[key] = entry
                        end

                        local sKey

                        if isTruth then
                            sKey = 'gt'
                        else
                            sKey = 'im'
                        end

                        if entry[sKey] then
                            print('Warning: Duplicate entry \'',p,'\'')
                        else
                            entry[sKey] = z .. '/' .. p
                        end

                    end
                end
            end
        end 

    end

    self.m_train = examples

    print('Found',table.sz(examples),'training examples!')

end

function data_loader:allocate_holdout()

    torch.manualSeed(42)

    nt = { }
    self.m_test = { }

    for k,v in pairs(self.m_train) do

        if torch.uniform() < 0.1 then
            self.m_test[k] = v
        else
            nt[k] = v
        end

    end

    self.m_train = nt

    print('Training Set',table.sz(self.m_train))

    print('Holdout Set',table.sz(self.m_test))

end





