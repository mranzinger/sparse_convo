require 'torch'
require 'lfs'
require 'utils'
require 'image'

package.path = '/home/mranzinger/dev/torch-transition/lua/?.lua;' .. package.path

require 'bw_deformation'
require 'gaussian_blur_deformation'
require 'gaussian_noise_deformation'
require 'compression_deformation'
require 'pca_augmentation'

local data_loader = torch.class('DataLoader')

function data_loader:__init()

    local prob = 0.13

    self.m_deforms = { 
        { prob, bw_deformation },
        { prob, gaussian_blur_deformation },
        { prob, gaussian_noise_deformation },
        { prob, compression_deformation },
        { prob, pca_augmentation }
    }

end

function data_loader:find_examples(a_dsDir)

    assert(paths.dirp(a_dsDir), 'The dataset directory doesn\'t exist!')

    print('Scanning',a_dsDir,'for training examples.')

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

    print('Found',table.sz(examples),'training examples!')

    self:allocate_holdout(examples)

end

function data_loader:allocate_holdout(examples)

    torch.manualSeed(42)

    self.m_train = self.m_train or { }
    self.m_test = self.m_test or { }

    for k,v in pairs(examples) do

        if torch.uniform() < 0.1 then
            table.insert(self.m_test, v)
        else
            table.insert(self.m_train, v)
        end

    end

    print('Training Set', #self.m_train)

    print('Holdout Set', #self.m_test)

end

function data_loader:study_net(a_net)

    --local dsX = 1
    --local dsY = 1

    local dsPath = nn.Sequential()

    --local dsXPath = { }
    --local dsYPath = { }

    for i=1,a_net:size() do
        local l = a_net:get(i) 

        if torch.type(l) == 'cudnn.SpatialMaxPooling' then
            dsPath:add(nn.SpatialMaxPooling(l.kW, l.kH,
                                            l.dW, l.dH))
            --dsX = dsX * l.dW
            --table.insert(dsXPath, l.dW)

            --dsY = dsY * l.dH
            --table.insert(dsYPath, l.dH)
        end
    end

    --print('Downsampling:',dsX,'x',dsY)

    --self.m_dsX = dsXPath
    --self.m_dsY = dsYPath
    self.m_ds = dsPath

end

function data_loader:finalize()

    -- TODO: Calculate mean and standard deviation

end

function data_loader:get_train_example()

    local idx = (torch.random() % #self.m_train) + 1

    local ex = self.m_train[idx]

    assert(ex)

    return self:get_example(ex)

end

function data_loader:num_test_examples()
    return #self.m_test
end

function data_loader:get_test_example(a_which)

    local ex = self.m_test[a_which]

    assert(ex)

    return self:get_example(ex)

end

function data_loader:get_example(a_ex)

    local imgPath = a_ex.im
    local gtPath = a_ex.gt

    assert(imgPath and gtPath)

    local img = image.load(imgPath)
    local gt = image.load(gtPath)

    assert(img and gt)
    assert(img:dim() == 3, 'The image has the wrong dimensionality!')
    assert(img:dim() == gt:dim(), 'The image and truth were not the same dimension!')
    for i=1,img:dim() do
        assert(img:size(i) == gt:size(i), 'The image and truth weren\'t the same size.')
    end

    data = self:prepare_data(img, imgPath)
    labels = self:prepare_gt(gt)
     
    return data, labels

end

function data_loader:prepare_data(img, imgId)

end

function data_loader:apply_deforms(img, imgId)

    for i=1, #self.m_deforms do

        local df = self.m_deforms[i]

        if torch.uniform() < df[1] then

            df[2](img, imgId)

        end
    end

end

function data_loader:prepare_gt(gt)
    
    -- First: Convert the GT image into a binary mask
    gt:apply(
        function(v)
            if v == 1.0 then
                return 0.0
            else
                return 1.0
            end
        end
    )

    gt = torch.max(gt, 1)

    -- Next: Resize the label image by the downsample
    --local imH = img:size(2)
    --local imW = img:size(3)

    --for i=1,#self.m_dsX do
    --    local sx = self.m_dsX[i]
    --    local sy = self.m_dsY[i]

    --    imH = math.ceil(imH / sy)
    --    imW = math.ceil(imW / sx)
    --end
    gt = self.m_ds:forward(gt)

    return gt

end

function data_loader:calc_mean_std()


end


















