import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, bbox_iou

def parse_annotation(ann_dir, img_dir, labels):
    all_imgs = []
    for ann in os.listdir(ann_dir):
        img = {'object':[], 'filename': img_dir + ann[:-4] + '.bmp'}     
        
        filename = ann_dir + ann
        
        image  = cv2.imread(img['filename'])
        element_height, element_width, _ = image.shape   
        img['width'] = int(element_width)
        img['height'] = int(element_height)
   

        root = ET.parse(ann_dir + ann)
        for name in ['figureRegion','formulaRegion','tableRegion']:
            p = root.findall(name)
            for line in p:
                coords = line.getchildren()
                line = coords[0].attrib['points']
                obj = {}
                obj['name'] = name
                
                line = line.split(' ')
                xy = line[0]
                x,y = xy.split(',')
                obj['xmin'], obj['ymin'] = int(x), int(y)
                
                xy = line[-1]
                x,y= xy.split(',')
                obj['xmax'], obj['ymax']= int(x), int(y)  
                if obj['name'] != 'figureRegion' or obj['ymax'] - obj['ymin'] > 15:
                    img['object'] += [obj]              

        if len(img['object']) > 0 :
            all_imgs += [img]

    return all_imgs


class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm


        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1), lightness=(0.9, 1.1)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 5), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        #iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        #iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        y_batch = [[], [], []]
        y_batch[0] = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))
        y_batch[1] = np.zeros((r_bound - l_bound, self.config['GRID_H']*2,  self.config['GRID_W']*2, self.config['BOX'], 4+1+len(self.config['LABELS'])))
        y_batch[2] = np.zeros((r_bound - l_bound, self.config['GRID_H']*4,  self.config['GRID_W']*4, self.config['BOX'], 4+1+len(self.config['LABELS'])))         
        # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            true_box_index = [0,0,0]
            
            for obj in all_objs:
                temphdw = obj['ymax'] - obj['ymin']
                ignore = False
                for output_i in range(3):
                    #y[0] 不检测数学公式以及直线  
                    # 形状过于奇怪也不检测        
                    output_scale = 2 ** output_i
                    if output_i == 0 and (obj['name'] == self.config['LABELS'][-1]):
                        continue
                         
                    if output_i == 2 and (obj['name'] != self.config['LABELS'][-1]):
                        continue       
                        
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                        center_x = .5*(obj['xmin'] + obj['xmax'])
                        center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W']/ output_scale )
                        center_y = .5*(obj['ymin'] + obj['ymax'])
                        center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H']/output_scale )

                        grid_x = int(np.floor(center_x))
                        grid_y = int(np.floor(center_y))

                        if grid_x < self.config['GRID_W']*output_scale and grid_y < self.config['GRID_H']*output_scale:
                            obj_indx  = self.config['LABELS'].index(obj['name'])
                            center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']/output_scale ) 
                            center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']/output_scale ) 
                            # 若此处以及存在标注，只标注面积较大者
                            if center_w * center_h < y_batch[output_i][instance_count, grid_y, grid_x, 0, 4  ]:
                                continue

                            box = [center_x-grid_x, center_y-grid_y, center_w, center_h]
                            minx = max(center_x - center_w/2,0)
                            maxx = min(minx + center_w, self.config['GRID_W']*output_scale)
                            miny = max(center_y - center_h/2,0)
                            maxy = min(miny + center_h, self.config['GRID_H']*output_scale)   
                            minx, maxx, miny, maxy = int(minx), int(maxx), int(miny), int(maxy) 
                            if maxy-miny > 1 and maxx-minx > 1:
                                temp = np.ones((maxy-miny, maxx-minx,  self.config['BOX']))
                                temp = y_batch[output_i][instance_count, miny:maxy, minx:maxx, :, 4  ] < temp
                                y_batch[output_i][instance_count, miny:maxy, minx:maxx, :, 4  ] = y_batch[output_i][instance_count, miny:maxy, minx:maxx, :, 4  ] - temp
                            y_batch[output_i][instance_count, grid_y, grid_x, :, 0:4] = self.config['BOX'] * [box]
                            y_batch[output_i][instance_count, grid_y, grid_x, :, 4  ] = self.config['BOX'] * [max(center_w*center_h, 1)]
                            y_batch[output_i][instance_count, grid_y, grid_x, :, 5+obj_indx] = 1.0


                            
            # assign input image to x_batch

            x_batch[instance_count] = self.norm(img)

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        return x_batch, y_batch


    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        '''
        if w > h:
            temph = int(1.3*w)
            tempimg = 254*np.ones((temph,w,c))
            tempimg[:h] = image
            image = tempimg
            h = temph
        '''
                
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]
            '''
            newscale = np.random.uniform() / 10. + 1.
            img = 255*np.ones( (int(h*newscale), int(w*newscale), 3) )
            img[:h,:w] = image
            image = img
            h, w, c = image.shape
            '''
            

            ### flip the image
            flip = False#np.random.random() > 0.6
            if flip : 
                image = cv2.flip(image, 1)
            '''    
            try:    
                image = self.aug_pipe.augment_image(image)      
            except:
                print(image_name)
            '''
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_W'], self.config['IMAGE_H']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = obj[attr] * float(self.config['IMAGE_W']) / w
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = obj[attr] * float(self.config['IMAGE_H']) / h
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip :
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
            if obj['ymax'] -  obj['ymin'] < 1:
                obj['ymax'] =  obj['ymin'] + 1                
            if jitter and np.random.random() > 0.9 and obj['name'] != self.config['LABELS'][-1] :
                x1,y1,x2,y2 = int(round(obj['xmin'] - 1 )),int(round(obj['ymin'] -1)),int(round(obj['xmax']+1 )),int(round(obj['ymax']+1 ))
                temp = image[y1 :y2,x1:x2]
                image[y1 :y2,x1:x2]= cv2.flip(temp, 1)
                             

        return image, all_objs
