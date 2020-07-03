#############################################################
#   Author : Rachid LADJOUZI                                #
#   Script : For Calibration Camera                         #
#   Input File : Must be locate in Folder Data              #
#   and be have name: Reference_Quality_Calib_Input.mp4     #
#   You can change Extention in Script                      #
#############################################################

import numpy as np
import cv2
import glob
import yaml
import sys

def RecupCalibrationParameters(filePathInput, filePathOutput, configPathOutput):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    found = 0
    gray2 = None

    cap = cv2.VideoCapture(filePathInput)                   #Input
    ret, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if ret == True:
        (w, h) = (h, w) = img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filePathOutput, fourcc, fps, (w, h))

    while(len(objpoints) < 50):  # Here, 10 can be changed to whatever number you like to choose
        # Capture frame-by-frame
        for i in range (0,10):
            ret, img = cap.read()

        if ret != True:
            break

        # img = image_resize(img, height = 1000)
        (h, w) = img.shape[:2]
        print("khra ", (h, w))

        cv2.imshow("Image", img)

        # Our operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gray2 = gray.copy()
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("ret true ")
            objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (6,9), corners2, ret)
            found += 1
            cv2.imshow('img', img)

        out.write(img)
    print("Number of images used for calibration: ", found)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray2.shape[::-1], None, None)

    # transform the matrix and distortion coefficients to writable lists
    data = {'camera_matrix': np.asarray(mtx).tolist(),
            'dist_coeff': np.asarray(dist).tolist()}

    # and save it to a file
    with open(configPathOutput, "w") as f:
        yaml.dump(data, f)



### MAIN ###
if(len(sys.argv) < 2):
    print("Plz enter reference Camera")
    exit()

refCamera = sys.argv[1]
filePathInput =  'Data/InputVideo/'+refCamera+'_Calib_Input.mp4'
filePathOutput =  'Data/OutputVideo/'+refCamera+'_Calib_Output.mp4'
configPathOutput = 'Data/OutputConfig/'+refCamera+'_CalibMatrix_Output.yaml'

RecupCalibrationParameters(filePathInput, filePathOutput, configPathOutput)