function [all_path, categ_info] = get_dataset_path(dataset_name, dataset_hier)
% Function Description : get all file path of images in the given dataset.
%
% [path_lst, hier_info] = get_dataset_path(dataset_name, dataset_hier, prefix)
% 
% Note : The function is location independent, 
%           but you need to confirm the dataset are all placed in the
%           parent directory of current working directory, 
%           where you execute the caller function.
% 
% Input :
%   dataset_name : a string, which record the absoulte path of dataset.
%   dataset_hier : a string, which record the relative path of the using dataset.
%
% Output :
%   path_lst : a list, which contains all path of image,
%                      you can read it via single iteration.  
%
%   categ_info : a functor (function object), which dynamic calculate 
%                      the hierachical information of dataset.
%
% Example: 
%   [all_path, categ_info] = get_dataset_path('dataset','train_set/HR_img');
%
% Contribution :
%   The consideration of dataset hierachy is unnecessary, 
%       we can access the dataset without any prior-mat file.
%
% License :  Apache License 2.0
% 
% Author : Joseph, 2020, 07, 03.
%

%% Note : the filesep is cross-platform, so the dataset_hier should use filesep
%%            to seperate the path, ex. 'CASIA\HR' in windows, 
%%                                      'CASIA/HR' in Linux.

    %% (1 Process from current path
    %     1. Jump to parent dir
    path_lst = regexp(pwd, filesep, 'split');    
    path_lst{end} = strcat(dataset_name, filesep, dataset_hier); 
    data_dir = strjoin(path_lst, filesep);
    %     2. filter the trivial ('.', '..') dir name 
    dir_lst = dir(data_dir);
    unsort_filename = {dir_lst.name};
    unsort_filename = { unsort_filename{3:end} };
    %     3. get filename prefix and extension
    filename_lst = split(unsort_filename{1}, '.');
    prefix = regexp(filename_lst{1}, '[a-zA-Z]*', 'match');
    ext = filename_lst{2};
    
    [filename_rnk, ~] = regexp(unsort_filename, '\d*','match', 'split');
    %     4. convert the container from cell to double arr
    categ = length(filename_rnk{1});
    len = length(filename_rnk);
    path_grp = zeros(len, categ);
    categ_info = zeros(1, categ);
    for idx=1:categ
        for jdx=1:len
            tmp = filename_rnk{jdx};
            path_grp(jdx, idx) = [str2num(tmp{idx})];
        end
        categ_info(idx) = max(path_grp(:, idx));
    end
    categ_info = categ_info(1:end-1);
    
    %     5. sorting the path by cross_comp
    for idx=1:len-1
        min = idx;
        for jdx=idx+1:len
            tag = cross_comp(path_grp(min, :), path_grp(jdx, :), categ);
            if(tag)
                min = jdx;
            end
        end
        tmp = path_grp(min, :);
        path_grp(min, :) = path_grp(idx, :);
        path_grp(idx, :) = tmp;
        
    end
    
    %     6. design hierachical template with extension dot
    template = '%d';
    for idx=1:categ-1
        template = strcat(template, '_%d');
    end
    template = strcat(template, '.');
    
    %     7. make all file path.
    all_path = cell(1, len);
    
    for idx=1:len
        raw_template = strcat(prefix, template, ext);
        filename_template = sprintf( raw_template{1}, path_grp(idx, :) );              
        all_path{idx} = strcat(data_dir, filesep, filename_template);
    end
    
end

function tag = cross_comp(val1, val2, categ)
    
    for idx=1:categ
        if val1(idx) > val2(idx)
            tag = 1;
            return;
        elseif val1(idx) < val2(idx)
            tag = 0;
            return;
        end
    end
    error('Duplicate filename error..');
end