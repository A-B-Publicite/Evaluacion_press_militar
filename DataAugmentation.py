import os
import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array 
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
DG_folder='DT_IMAGEN_INCORRECTO'
images_increased = 4

try:
    os.mkdir(DG_folder)
except:
    print("")
    
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)


data_path = "F:\EPN\QUINTO SEMESTRE\Inteligencia artificial\PROYECTO SEMESTRAL\INCORRECTO"
data_dir_list = os.listdir(data_path)


width_shape, height_shape = 200, 200

i=0
num_images=0
for image_file in data_dir_list:
    img_list=os.listdir(data_path)

    img_path = data_path + '/'+ image_file

    imge=load_img(img_path)
    
    imge=cv2.resize(image.img_to_array(imge), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
    x= imge/255
    x=np.expand_dims(x,axis=0)
    t=1
    for output_batch in train_datagen.flow(x,batch_size=1):
        a=image.img_to_array(output_batch[0])
        imagen=output_batch[0,:,:]*255
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(DG_folder+"/%i%i.jpg"%(i,t), imgfinal) 
        t+=1
        
        num_images+=1
        if t>images_increased:
            break
    i+=1
    
print("images generated",num_images)