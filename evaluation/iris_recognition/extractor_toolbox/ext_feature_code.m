% IV. Feature extraction :
%==========================================================================
function feaCell = ext_feature_code(feaType, irisPolar, maskCell, user_def_params, db_info, iris_path_template)
    %% Get database information 
    IOM_statistics = db_info.statistics;
    
    %% Get user defined parameters
    show_process = user_def_params.show_process;
    limitation = user_def_params.limitation;
    chkLis = user_def_params.chkLis;

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

    feaCell = cell(size(IOM_statistics));
    [col, row] = size(IOM_statistics);
    for idx=1:col-limitation
        for jdx=1:row
            len = IOM_statistics(idx, jdx);
            cls_fea = cell(len);

            for zdx=1:len
                % Access the cell component of polar coordinate
                %   and iris mask information along element by element scaning {zdx}.
                iris_polar = irisPolar{idx, jdx}{zdx};
                mask_cell = maskCell{idx, jdx}{zdx};
                % Feature extraction.
                iris_fea = fea_extor(iris_polar, params, mask_cell);
                cls_fea{zdx} = iris_fea;

                if show_process && (idx==1) && (jdx==1) && (chkLis == zdx)
                    Visual_Result(mask_cell, iris_fea,...
                        'Polar coordinate iris img', 'Corresponding mask');
                    pause;
                end
                
            end
            feaCell{idx, jdx} = cls_fea;

        end
    end
    
end