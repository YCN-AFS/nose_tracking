import cv2
import time
import threading
#import pyautogui
import numpy as np
import mediapipe as mp
from mouse import *
from gaze_tracking import GazeTracking

pyautogui.FAILSAFE = False





gaze = GazeTracking()
mp_face_mesh = mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh()
mp_drawing_styles = mp.solutions.drawing_styles

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

cap = cv2.VideoCapture(0)
pTime = 0
wait_time = time.time()



while cap.isOpened():
    #Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    success, frame = cap.read()
     
 
   
    
    #Eyes tracking
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    #Flip and convert image from BGR to RGB
    frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)


    frame.flags.writeable = False

    results = face_mesh.process(frame)

    frame.flags.writeable = True

    #Convert image from RGB to BGR
    frame = frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark): 
                if id in (33, 263, 1, 61, 291, 199):
                    if id == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z *3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    #Get 2D and 3D Coordinates
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])

            #Convert it to numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            #The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0,0,1]])
            #The distortion parameters
            dist_matrix = np.zeros((4,1), dtype=np.float64)

            #Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            #Get rotation matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            #Get anglesq
            angles, mtxR, mtxQ, Qx, Qy, Qz =cv2.RQDecomp3x3(rmat)

            #Get the y rotation degree
            x = angles[0] * 360 
            y = angles[1] * 360
            z = angles[2] * 360



            #Display nose direction
            noce_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] -x * 10))

            cv2.line(frame, p1, p2, (0, 255, 255), 2)

    #Add text on the image
    

    cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)


    #click
    if gaze.is_left():
        pyautogui.click()
    if gaze.is_right():
        pyautogui.rightClick()
     #mouse scroll
   
    

    threa_scroll = threading.Thread(target=scroll_mose, args=(x,))
  
    if abs(x) > 22 and time.time() - wait_time > 0.5:
        wait_time = time.time()
        pyautogui.scroll(2)
        #threa_scroll.start()


    #Run multi-threaded mouse
    mouse_thea = threading.Thread(target = move_mouse, args=(x,y,))
    if not mouse_thea.is_alive():
        mouse_thea.start()

    
   

    
    cv2.putText(frame, "FPS: " + str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255, 0), 2)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Show video
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Dieu khien con tro chuot", frame)
   
    
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


threa_scroll.join()
mouse_thea.join()
cap.release()
cv2.destroyAllWindows()

