%% iris feature extractor :

%% TODO : utils/ iter_tool.
function iris_feature_extractor(dataset_name, dataset_hier, ld_bound, save_code_path)
    addpath('./utils')    
    addpath('./extractor_toolbox');
    
    % Setting the path template for reading :
    tic;
    % I. Iris segmentation && Polar Coordinate transform 
    %==========================================================================
    bound_path_cell = {};
    if ~isempty(ld_bound)
        [bound_path_cell, ~] = get_dataset_path(dataset_name, ld_bound);
    end
    
    %PolarType = input('Please type the num (Crt2Polar/1 ; Crt2LogPol/2(by default)) for choice method.. \n');
    PolarType = 1;
    
    %@ 1.get function handler of transform method..
    switch PolarType
        case 1
            transform = @Cart2PolarFromCircle;
        case 2
            transform = @Cart2LogPolarFromCircle;
        otherwise
            error('No such method : Polar Coordinate transform..\n');
    end
    
    %@ 2.read iris (image & information) and transform into polar coordinate.
    [img_path_cell, categ_info] = get_dataset_path(dataset_name, dataset_hier);
    len = length(img_path_cell);
    categ_info = [1, categ_info];
    irisPolar = cell(categ_info);
    counter = zeros(categ_info);
    
    cnt = 1; buff_ptr = 1; tmp_buff = {};
    for idx=1:len

        cls_tag = get_cls_tag(img_path_cell{idx});
        %% write the iris polar info from temporary buffer.
        if (cls_tag(1) ~= cnt)
            irisPolar{cnt} = tmp_buff;
            tmp_buff = {};
            cnt = cnt + 1;
            counter(cnt) = buff_ptr;
            buff_ptr = 1;
        end
        
        %@@ 1. get iris image :
        img = imread(img_path_cell{idx});
        img_siz = size(img);
        channel = length(img_siz);

        %@@@ 2. image channel issue :
        if channel >= 2
            switch channel
                case 3
                    img = rgb2gray(img);
                case 2
                    img = img;
                otherwise
                    error('Do not support hyper-channels image (greater than 3).');
            end
        else          
            error('The input image is invalide, which channel less then 2.');
        end

        %@@ 3. extract circular info along the class (cls%idx_%jdx_%zdx) :
        if  isempty(ld_bound)
            disp(img_path_cell{idx});
            cirinfo = irisSegNear(img);
            iris_circle = cirinfo(1:3);   
            pupil_circle = cirinfo(4:6);
        else
            % record (x, y, radius) of 2 boundary.
            tmp = csvread(bound_path_cell{idx});
            iris_circle = tmp(1:3);   
            pupil_circle = tmp(4:6); 
            %% draw both circle..
        end

        %@@@ 4. image size issue :
        div = 1;
        if (img_siz(1) > 480) || (img_siz(2) > 640)
            error('The image size should equal or less than [ 480 x 640 ] for h x w.');
        else
            h_div = 480 / img_siz(1);
            w_div = 640 / img_siz(2);

            if h_div ~= w_div
                error('The shrink percentage should be equal.');
            end
            div = h_div;
        end
        cirInfo = [iris_circle/div, pupil_circle/div];

        %@@ 5. polar coordinate transformation :
        % compress multi-dim data into class dim {just one dim}.
        tmp_buff{buff_ptr} = transform(img, cirInfo);  
        
        buff_ptr = buff_ptr + 1;

    end
    irisPolar{cnt} = tmp_buff;
    counter(cnt) = buff_ptr;
    %% pass.. continue further code..
    toc;
    fprintf('Polar Coordinate transform done..\n\n');
    %pause;
    %clc;
    
    % III. Mask generation
    %==========================================================================
    %@ 1.Get mask generator.
    %maskType = input('choice mask type, 1 -> Rule based ; 2 -> Gaussian Mixture Models\n');
    maskType = 1;
    params = [];
    switch maskType
        case 1
            maskGeneator = @createRuleBasedMask;
        case 2
            maskGeneator = @IrisMaskEstiGaborGMM;
            load('./load/GmmModel_7Gb.mat');      % load pre-training model 
            params.bayesS = bayesS;
            params.GbPar = GbPar;
        otherwise
            error('No such type mask generator..');
    end
    
    %@ 2.Generate mask for polar coordinate images.
    maskCell = cell(categ_info);
    len = length(irisPolar);
    for idx=1:len
        polar_iris = irisPolar{idx};
        siz = length(irisPolar{idx});
        msk_lst = cell(1, siz);
        for jdx=1:siz
            
            if isempty(params)
                msk_lst{jdx} = maskGeneator(polar_iris{jdx});
            else
                msk_lst{jdx} = maskGeneator(polar_iris{jdx}, params);
            end
            
        end
        maskCell{idx} = msk_lst;
    end
        
    fprintf('Mask generation done..\n\n');
    %pause;
    %clc;
    
    % IV. Feature extraction :
    %==========================================================================
    %feaType = input('Please type the num (compare/0 ; LiborMasekIrisCode/1 ; KhalidIrisCode/2(by default)) for choice method.. \n');
    feaType = 2;
    switch feaType
        case 1
            % Set Path, should import iris_lib/irlib/iris/Normal_encoding/encode
            fea_extor = @LiborMasekIrisCode;  
            % input parameters..Libor
            params = {};
            params.shift_range = [-20:2:20];
            params.method = 'LM';
            params.nscales = 1;
            params.minWaveLength = 18;
            params.mult = 1;
            params.sigmaOnf = 0.5;

        case 2
            fea_extor = @KhalidIrisCode;
            % input parameters..Kahlid
            params = {};
            params.resize_height = 30;  %%30
            params.resize_width = 360;  %%360
            params.sigma_x = 3;
            params.sigma_y = 3;

        otherwise
            error('No such iris code method..');
    end
    
    feaCell = cell(categ_info);
    
    for idx=1:len
        iris_polar = irisPolar{idx};
        mask_lst = maskCell{idx};
        siz = length(iris_polar);
        fea_lst = cell(1, siz);
        
        for jdx=1:siz
            % Feature extraction.
            fea_lst{jdx} = fea_extor(iris_polar{jdx}, params, mask_lst{jdx});
            
        end
        feaCell{idx} = fea_lst;
    end
    
    fprintf('Feature extraction done..\n\n');
    %pause;
    %clc;
    
    % setting save path & name of mat file.
    save(save_code_path, ...
        'irisPolar', 'maskCell', 'feaCell');
    
end

function cls_tag = get_cls_tag(path_str)
    path_lst = split(path_str, filesep);
    [tmp, ~] = regexp(path_lst{end}, '\d*','match', 'split');
    len = length(tmp);
    cls_tag = zeros(1, len);
    % from string to number.
    for idx=1:len 
        cls_tag(idx) = str2num(tmp{idx});
    end
end
