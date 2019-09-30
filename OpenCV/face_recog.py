# Recognising face using some algorithm like Logistic Regression, KNN, SVM etc.


# Step 1. Load the training data (numpy arrays of all the person)
#               x-values are stored in the numpy arrays
#               y-values we need to assign fo reach person
# Step 2. Read a video stream using opencv
# Step 3. Extract faces out of it
# Step 4. Use KNN to find the prediction of face (int)
# step 5. Map the predicted idto name of the user
# Step 6. Display the prediction on the screen - bounding box and name

import cv2
import numpy as np
import os

############## KNN Code ###############


def distance(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(train,test,K=5):
    dist = []
    m = train.shape[0]

    for i in range(m):
        
        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:K]
    # Pick the Nearest/First K points
    labels = np.array(dk)[:, -1]
    
    output = np.unique(labels, return_counts=True)
    # print(new_vals)

    index = np.argmax(output[1])
    

    # print(index)
    # print(pred)
    # print(vals)
    # return vals
    return output[0][index]
######################################


# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './Data/'

face_data = []  # x-values
labels = []  # y-values
classid = 0  # Labels for the files
names = {}  # mapping between classid and name

# Data Preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        #Create a mapping between class id and name
        
        names[classid] = fx[:-4]
        
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        
        #Create labels for the class
        target = classid*np.ones((data_item.shape[0],))
        classid += 1
        labels.append(target)
        
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

#Testing

while True:
    
    ret,frame = cap.read()
    
    if ret==False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        offset = 10 
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Predicted label
        out = knn(train_set,face_section.flatten())

        #Display on the screen the name and rectangle 
        
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow('Frame',frame)

    keypressed = cv2.waitKey(1) & 0xff
    if keypressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()