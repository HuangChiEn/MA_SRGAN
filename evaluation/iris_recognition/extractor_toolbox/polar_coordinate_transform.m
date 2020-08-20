% II. Polar Coordinate transform 
%==========================================================================
function irisPolar = polar_coordinate_transform(PolarType, user_def_params, db_info, iris_path_template)
    %% Get database information 
    IOM_statistics = db_info.statistics;
    iom_cirinfo = db_info.boundary_info;

    %% Get user defined parameters
    show_process = user_def_params.show_process;
    limitation = user_def_params.limitation;
    chkLis = user_def_params.chkLis;
    
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
    irisPolar = cell(111, 3);
    [col, row] = size(IOM_statistics);
    for idx=1:col-limitation
        for jdx=1:row
            len = IOM_statistics(idx, jdx);
            cls_polar = cell(len);  %% preallocate storage for ploar_img of class.
                       
            for zdx=1:len
                %@@ 1.extract circular info along the class (cls%idx_%jdx_%zdx) :
                % record (x, y, radius) of 2 boundary.
                iris_circle = iom_cirinfo{idx, jdx}(zdx, 1:3);  
                pupil_circle = iom_cirinfo{idx, jdx}(zdx, 4:6); 

                %@@ 2.get iris image.
                iris_path = sprintf(iris_path_template, idx, jdx, zdx);
                img = imread(iris_path);
                img_siz = size(img);
                channel = length(img_siz);
                
                %@@ 3.preprocess the input images :
                %@@@ 1) image channel issue-
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
                
                %@@@ 2) image size issue-
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
                
                %@@ 4.polar coordinate transformation.
                polar_iris = transform(img, cirInfo);
                cls_polar{zdx} = polar_iris;

                %@@ 5.plot result.
                if show_process && (idx==1) && (jdx==1) && (chkLis == zdx)
                    Visual_Result(img, polar_iris, ...
                        'Raw iris img', 'Polar coordinate via (Cart2PolarFromCircle)');
                end

            end

            %@@ 6.store the polar coordinate of same class img.
            irisPolar{idx, jdx} = cls_polar;
        end
    end
end
