import numpy as np


def get_rotation_degree(results):
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
            return (x, y,)