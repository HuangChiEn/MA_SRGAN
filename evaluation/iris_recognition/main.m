%% Main controller --                                                      %%
%% Description : Define the parameters &&                                  %%
%%               Call the function to complete iris recognition            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1). Recognition parameters setting -
dataset_name = 'iris_data';
dataset_hier = 'IOM\HR';
ld_bound = 'IOM\boundaries';
save_code_path = 'iris_information\HR_iris_info\tstdel.mat';
save_hd_path = 'iris_information\HD_value\HR\tstdel.mat';
build_in_seg = false;

%% 2). Iris Recognition :
%==================================================================================
iris_feature_extractor(dataset_name, dataset_hier, ld_bound, save_code_path);
disp(['fea extra over']);
pause;

results = largeScale_comparison(save_code_path, save_hd_path);

%clear;
%% ROC curves plotting : 
%==================================================================================
% addpath('visual_tool');
% load_cell = {'iris_information\HD_value\HR\exp_CAhr_001.mat', ...
%                 'iris_information\HD_value\HR\exp_002_woMsk', ...
%                     'iris_information\HD_value\HR\exp_CAsrLab_001', ...
%                         'iris_information\HD_value\LR\exp_CAlr_001'};
% legnd_str_lst = ['', ''];
% plot_params = {'--r', '--b', '-m', '-k'};
% plot_roc_curves(load_cell, legnd_str_lst, plot_params);
% pause;
% clc;
