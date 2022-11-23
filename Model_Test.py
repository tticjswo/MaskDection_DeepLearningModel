import os, re, glob
import cv2
import numpy as np
import shutil
from google.colab.patches import cv2_imshow
from keras.models import load_model
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

def Dataization(img_path):
    img = cv2.imread(img_path)
    return (img / 256)


src = []
name = []
test = []
image_dir = 'test_data_3/'
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):
        src.append(image_dir + file)
        name.append(file)

        img = cv2.imread(image_dir + file)

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(305,305), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue
            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            if (x2 >= w or y2 >= h):
                continue

            face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (200, 200))
        test.append(face/256)
        # cv2.imwrite(image_dir +'test/'+file,face)


test = np.array(test)
print(test.shape)
model = load_model('modelv5')
predict = model.predict(test)
print(predict.shape)
print("ImageName : , Predict : [mask, nomask]")

correct_Predict = 0
for i in range(len(test)):
    if((predict[i][0]).item() > (predict[i][1]).item() ) :
      if name[i][-8:] =='mask.jpg' :
        correct_Predict= correct_Predict +1
      print(name[i] + " : ,      Predict : [mask]")
    else :
      if name[i][-8:] !='mask.jpg' :
        correct_Predict= correct_Predict +1
      print(name[i] + " : ,       Predict : [no_mask]")

print("precsion = " + str(len(test)) + "/" + str(correct_Predict)+ '=' + str(correct_Predict/len(test) * 100))

    # print(name[i] + "     : ,           Predict : " + str(predict[i]))