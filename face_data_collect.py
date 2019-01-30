import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
filename = input("Enter the name of the person : ")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_section = frame

    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key = lambda f:f[2]*f[3])
    #print(faces)
    
    # Pict the last face because it is the largest face according to area given by f[2]*f[3]
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #Extract or crop out the required face section
        offset = 10 #This is the padding around the face that we add
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip += 1    
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
 
    cv2.imshow("Video Frame",frame)
    cv2.imshow("Face Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break    

#Convert face list array to numpy array by flattening the pixel values of the image into 1xn form where n is the number of pixels
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+filename+".npy",face_data)
print("Data successffully saved at "+dataset_path+filename+".npy")

cap.release()
cv2.destroyAllWindows()