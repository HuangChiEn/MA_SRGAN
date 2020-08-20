% III. Mask generation
%==========================================================================
function maskCell = mask_generation(maskType, irisPolar, user_def_params, db_info, iris_path_template)
    %% Get database information 
    IOM_statistics = db_info.statistics;
    
    %% Get user defined parameters
    show_process = user_def_params.show_process;
    limitation = user_def_params.limitation;
    chkLis = user_def_params.chkLis;

    params = [];
    switch maskType
        case 1
            maskGeneator = @createRuleBasedMask;
        case 2
            maskGeneator = @IrisMaskEstiGaborGMM;
            load './load/GmmModel_7Gb.mat';      % load pre-traing model 
            params.bayesS = bayesS;
            params.GbPar = GbPar;
        otherwise
            error('No such type mask generator..');
    end

    %@ 2.Generate mask for polar coordinate images.
    maskCell = cell(111, 3);
    [col, row] = size(IOM_statistics);
    for idx=1:col-limitation
        for jdx=1:row
            len = IOM_statistics(idx, jdx);
            cls_mask = cell(len);

            for zdx=1:len
                polar_iris = irisPolar{idx, jdx}{zdx};

                if isempty(params)
                    mask = maskGeneator(polar_iris);
                else
                    mask = maskGeneator(polar_iris, params);
                end
                cls_mask{zdx} = mask;

                if show_process && (idx==1) && (jdx==1) && (chkLis == zdx)
                    Visual_Result(polar_iris, mask, ...
                        'Polar coordinate iris img', 'Corresponding mask');
                    pause;
                end
                
            end
            maskCell{idx, jdx} = cls_mask;
            
        end
    end
    
end