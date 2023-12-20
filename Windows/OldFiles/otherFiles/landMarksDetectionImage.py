import cv2
import mediapipe as mp

#Image
image= cv2.imread("/Users/roby/PycharmProjects/FER/person.jpg")
height, width, _ = image.shape
#print("Height,width", height, width)
rgb_image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  #converte schema colore da brg a rgb

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh #mediapipe
face_mesh = mp_face_mesh.FaceMesh() #load object

#Facial LandMarks
result=face_mesh.process(rgb_image)


# Lista dei punti corrispondenti alle sopracciglia, agli occhi, al naso e alla bocca
landmark_indices = [1,4,19,44,274,275,354,440]


'''
for facial_landmarks in result.multi_face_landmarks:
    for i in range (0,468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * width)
        y= int(pt.y * height)

        cv2.circle(image,(x,y),2,(100, 100, 0), -1) #disegna il punto sull'immagine, alle coordinate x,y con raggio 3 e di colore rgb e -1 per riempire il puntino con i colore
        #cv2.putText(image, str(i),(x,y),0,1,(0,0,0))
'''


# Itera sui punti facciali rilevati
for facial_landmarks in result.multi_face_landmarks:
    # Itera sui punti di interesse
    for index in landmark_indices:
        pt = facial_landmarks.landmark[index]
        x = int(pt.x * width)
        y = int(pt.y * height)

        # Disegna un cerchio per ogni punto
        cv2.circle(image, (x, y), 2, (100, 100, 0), -1)

# Visualizza l'immagine con i punti ridotti
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()






cv2.imshow("Image",image)
#cv2.imshow("rgb Image",rgb_image)
cv2.waitKey(0)
