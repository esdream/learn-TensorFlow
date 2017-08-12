
# coding: utf-8

# In[1]:

import os
from PIL import Image


# In[12]:

def split_green_band(classes_path):
    for img_dir in os.listdir(classes_path):
        if(os.path.isdir(img_dir)):
            g_band_dir = os.path.join(classes_path, '{}_{}'.format(img_dir, 'g_band'))
            if not os.path.exists(g_band_dir):
                os.makedirs(g_band_dir)
                
                for img_file in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_file)
                    img = Image.open(img_path).resize((32, 32))
                    r, g, b = img.split()

                    img_file, suffix = os.path.splitext(img_file)
                    g.save(os.path.join(g_band_dir, '{}_{}.jpg'.format(img_file, 'g_band')))
            else:
                print('Dir is existed!')


# In[13]:

split_green_band('.')

