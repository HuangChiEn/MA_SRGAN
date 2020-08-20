ori_path = '../iris_data/IOM/SR/';
dirlst = dir(ori_path);
img_path = {dirlst.name};
img_path = {img_path{3:end}};
scal = 4;
len = length(img_path);
for idx=1:len
    filNam = img_path{idx};
    path = strcat(ori_path, filNam); 
    img = imread(path);
    [M, N] = size(img);
    img = imresize(img, [120, 160]);
    %img = img(1:4:M,1:4:N);
    
    new_path = strcat("../iris_data/IOM/LR/", filNam);
    imwrite(img, new_path);
end
    