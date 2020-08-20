function results = largeScale_comparison(iris_info_path, save_hd_path)
    load(iris_info_path);
    
    len = length(maskCell);
    siz = length(maskCell{1});
    iris_code = cell(len*siz, 1);
    iris_mask = cell(len*siz, 1);
    iris_label = zeros(len*siz, 1);  % record each [iris_code, iris_mask] with class label.
    cnt = 1;
    
    %% Accumulate the iris feature code with cross-class.
    for idx=1:len
        msk_lst = maskCell{idx};
        fea_lst = feaCell{idx};
        siz = length(msk_lst);
        
        for jdx=1:siz
            %  Read iris code and cooresponding mask, 
            %       confirm the data type by explicit declartion.
            iris_mask{cnt, 1} = logical(msk_lst{jdx});
            iris_code{cnt, 1} = double(fea_lst{jdx});
            iris_label(cnt, 1) = idx;
            cnt = cnt + 1;
        end
    end  
    
    %% For the same resolution comparison.. 
    results = runExpwPredTemplateMask_LargeScale(struct('shift_range', [-20:2:20]), ...
                                        iris_code, iris_mask, iris_label, 30,...
                                        save_hd_path);
end

