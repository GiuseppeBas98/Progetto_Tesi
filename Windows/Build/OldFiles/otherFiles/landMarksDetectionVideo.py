import cv2
import mediapipe as mp


# Face Mesh
mp_face_mesh = mp.solutions.face_mesh #mediapipe
face_mesh = mp_face_mesh.FaceMesh() #load object

cap=cv2.VideoCapture("1.mp4")

while True:
    # Image
    ret, image = cap.read()
    height, width, _ = image.shape
    print("Height,width", height, width)
    rgb_image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  #converte schema colore da brg a rgb


    # Facial LandMarks
    result=face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range (0,468):
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * width)
            y= int(pt.y * height)

            cv2.circle(image,(x,y),2,(100, 100, 0), -1) #disegna il punto sull'immagine, alle coordinate x,y con raggio 3 e di colore rgb e -1 per riempire il puntino con i colore
            #cv2.putText(image, str(i),(x,y),0,1,(0,0,0))




    cv2.imshow("Image",image)
    #cv2.imshow("rgb Image",rgb_image)
    cv2.waitKey(1) #aspetta un msec e va al prossimo frame