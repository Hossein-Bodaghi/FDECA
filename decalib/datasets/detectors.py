# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

class FMP(object):
    def __init__(self, model_path):
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.model = vision.FaceLandmarker.create_from_options(options)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        img = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
        self.landmarks = self.model.detect(img)
        
        ldks = self.landmarks.face_landmarks
        ldks = ldks[0]
        kpt = np.zeros((len(ldks),2))
        face_landmarks_proto = []
        for idx in range(len(ldks)):
            face_landmarks = ldks[idx]
            a = landmark_pb2.NormalizedLandmark(x = face_landmarks.x,
                                                y = face_landmarks.y)
            kpt[idx] = solutions.drawing_utils._normalized_to_pixel_coordinates(a.y,
                        a.x, img.height,img.width)
                    
        if self.landmarks is None:
            return [0], 'kpt68'
        else:
            left = np.min(kpt[:,1]); right = np.max(kpt[:,1]); 
            top = np.min(kpt[:,0]); bottom = np.max(kpt[:,0])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'
        
    def detect(self, image, out=False):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        img = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
        self.landmarks = self.model.detect(img)
        if out:
            return self.landmarks
        

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'



