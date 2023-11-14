#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:55:17 2023

@author: Shiva_roshanravan
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import matplotlib.pyplot as plt
import numpy as np
import cv2 
import matplotlib.pyplot as plt

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

  return annotated_image

def masking_face(image, model_path):
    
    # plt.imshow(image)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
    # mask = np.zeros((image.height,image.width,3),dtype='uint8')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detection_result = detector.detect(image)    

    # mask = np.zeros_like(image)
    mask = np.zeros((image.height,image.width,3),dtype='uint8')
    
    oval_mask = np.copy(mask)
    oval_line = np.copy(mask)
    right_eye_mask = np.copy(mask)
    left_eye_mask = np.copy(mask)
    
    annotated_image = draw_landmarks_on_image(mask, detection_result)
    # plt.imshow(annotated_image)
    
    imgray = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('number of contours are: ', len(contours))
    try:
        cv2.drawContours(right_eye_mask, contours, 5, (255,255,255),-1)
    except:
        # cv2.drawContours(right_eye_mask, contours, 4, (255,255,255),-1)
        pass
    cv2.drawContours(oval_mask, contours, 1, (255,255,255),-1)
    cv2.drawContours(left_eye_mask, contours, 3, (255,255,255),-1)
    cv2.drawContours(oval_line, contours, 1, (255,255,255),5)
    
    oval_imgray = cv2.cvtColor(oval_mask, cv2.COLOR_RGB2GRAY)
    
    re_imgray = cv2.cvtColor(right_eye_mask, cv2.COLOR_RGB2GRAY)
    le_imgray = cv2.cvtColor(left_eye_mask, cv2.COLOR_RGB2GRAY)
    ret, thresh_ov = cv2.threshold(oval_imgray, 50, 255, 0)
    ret, thresh_re = cv2.threshold(re_imgray, 50, 255, 1)
    ret, thresh_le = cv2.threshold(le_imgray, 50, 255, 1)
    
    final_thr = thresh_ov*thresh_re*thresh_le
    final_thr = np.reshape(final_thr, (final_thr.shape[0],final_thr.shape[1],1))
    oval = np.reshape(thresh_ov, (thresh_ov.shape[0],thresh_ov.shape[1],1))
    return final_thr, oval, oval_line

#%%

# model_path = '/home/Shiva_roshanravan/Documents/FDECA/data/face_landmarker.task'
# img_path = '/home/Shiva_roshanravan/Documents/FDECA/TestSamples/examples/Partners/Reza.jpg'
# img = cv2.imread(img_path)
# a = masking_face(img, model_path)
# oval_line = a[2]
# plt.imshow(a[2])
