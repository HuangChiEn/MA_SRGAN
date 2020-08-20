import cv2
import os, random, math
import numpy as np
import multiprocessing.pool

class ImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h, img_c, batch_size):
        self.img_h = img_w # img weight, height switch, due to image_data_format..
        self.img_w = img_h
        self.img_c = img_c 
        self.batch_size = batch_size
        self.img_dirpath = img_dirpath                  # image dir path
        
#        self.img_dir = os.listdir(self.img_dirpath)     # images list
        with open( self.img_dirpath ) as f :
            lines = f.readlines()
            self.lines = lines[:]
            self.n = len(self.lines) # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        print( 'DATA SIZE :', self.n )
#        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
#        self.texts = []
    def center_score(self,distance,radius):
        return max(0,(1-distance/10))
#        return max(0,(1-distance/radius))
    
    ##  Calculate the IOU score, the center point contain highest score 1, 
    ##      and far from center point going down to 0.
    def push_value_2_score_map(self,pos,idx,loop_range=5):  
        ## initial score map with 0, and get boundary information (x, y, radius) "with half of it". 
        score_map = np.zeros([self.img_h,self.img_w,3])  ## 480 x 640 x 3(RGB)
        pos_x = int(float(pos[idx])/4) # 2
        pos_y = int(float(pos[idx+1])/4) # 2
        pos_r = int(float(pos[idx+2])/4) # 2
        
        for x in range(0,loop_range):#range(0,int(pos_r)):
            for y in range(0,loop_range):#range(0,int(pos_r)):
                dis = math.sqrt(x*x+y*y)
                score_xy=self.center_score(dis,pos_r)
                x_min = max(0,pos_x-x)
                y_min = max(0,pos_y-y)
                x_max = min(self.img_w-1,pos_x+x)
                y_max = min(self.img_h-1,pos_y+y)
                score_map[y_min,x_min,:]=[max(score_map[y_min,x_min,0],score_xy),  ## score_xy : score
                                          max(score_map[y_min,x_min,1],dis),       ## dis : distance from center to x_y
                                          pos_r]                                   ## pos_r : radius
                score_map[y_max,x_min,:]=[max(score_map[y_max,x_min,0],score_xy),
                                          max(score_map[y_max,x_min,1],dis),
                                          pos_r]
                score_map[y_max,x_max,:]=[max(score_map[y_max,x_max,0],score_xy),
                                          max(score_map[y_max,x_max,1],dis),
                                          pos_r]
                score_map[y_min,x_max,:]=[max(score_map[y_min,x_max,0],score_xy),
                                          max(score_map[y_min,x_max,1],dis),
                                          pos_r]
        return score_map
    
    def read_image_and_label(self, lines, index ):
        line = lines[index][:-1].split(' ')
        img_path = line[0]
        if (self.img_c==3):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32)
        img = img/255.0
        img = np.expand_dims(img,3)
        mask_path = line[1]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_w, self.img_h))
        # fix 
        mask = (mask != 0 ) * 1
        mask = mask.astype(np.float32)
        mask=np.expand_dims(mask,3)

        ## pos : 6 x 1 list.
        pos = line[2].split(',')     ## pos :=> pupil : (x, y, p_radius) ; iris : (x, y, i_radius). 
        pupil_score_map=self.push_value_2_score_map(pos,0)
        iris_score_map=self.push_value_2_score_map(pos,3,loop_range=15)  ## pos[3] : begin from iris info.
#        label /= 255.0
#        label = 1-label
        return np.squeeze(img, axis=-1), pupil_score_map, iris_score_map, mask
        
    def next_sample(self):      ## index max -> 0  
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.read_image_and_label( self.lines, self.indexes[self.cur_index] )
    
    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_c])
            Y_Pupil_Pos = np.zeros([self.batch_size, self.img_h, self.img_w, 3])
            Y_Iris_Pos = np.zeros([self.batch_size, self.img_h, self.img_w, 3])
            Y_Mask = np.zeros([self.batch_size, self.img_h, self.img_w, 1])
            pool = multiprocessing.pool.ThreadPool()
        
            # Second, build an index of the images
            # in the different class subfolders.
            results = []
            for i in range(self.batch_size):
                results.append( pool.apply_async( self.next_sample, ()) )
            
            for i, res in enumerate(results):
                X_data[i], Y_Pupil_Pos[i], Y_Iris_Pos[i], Y_Mask[i] = res.get()

            pool.close()
            pool.join()  
            yield (X_data, {'Pupil_Bondary':Y_Pupil_Pos,'Iris_Bondary':Y_Iris_Pos,'Mask':Y_Mask})
            
    def Get_Batch_Data(self):       
        X_data = np.zeros([self.batch_size, self.img_h, self.img_w, 1])
        Y_Pupil_Pos = np.zeros([self.batch_size, self.img_h, self.img_w, 3])
        Y_Iris_Pos = np.zeros([self.batch_size, self.img_h, self.img_w, 3])
        Y_Mask = np.zeros([self.batch_size, self.img_h, self.img_w, 1]) 

        pool = multiprocessing.pool.ThreadPool()
    
        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        for i in range(self.batch_size):
            results.append( pool.apply_async( self.next_sample, ()) )
#                img, label = self.next_sample()
#                img = img.T
#                img_data = np.expand_dims(img, -1)
#                label_data = np.expand_dims(label, -1)
        for i, res in enumerate(results):
            X_data[i], Y_Pupil_Pos[i], Y_Iris_Pos[i], Y_Mask[i] = res.get()

        pool.close()
        pool.join()

        return (X_data, {'Pupil_Bondary':Y_Pupil_Pos,'Iris_Bondary':Y_Iris_Pos,'Mask':Y_Mask})
