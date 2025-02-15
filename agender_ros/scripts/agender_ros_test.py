#!/usr/bin/env python
from __future__ import print_function

import os
import math
import time
import cv2 as cv
import numpy as np
from age_gender_ssrnet.SSRNET_model import SSR_net_general, SSR_net
from time import sleep

import roslib
roslib.load_manifest('agender_ros')
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import glob
import face_recognition as fr
from agender_ros_msgs.msg import PersonDescriptors, PeopleDescriptors

class agender_ros:

  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.rospack = rospkg.RosPack()
    self.models_path = self.rospack.get_path('agender_ros') + '/models/'
    self.people_path = self.rospack.get_path('agender_ros') + '/people/'
    self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw",Image,self.callback)
    self.image_pub = rospy.Publisher("/faces", Image, queue_size=1)
    self.descriptors_pub = rospy.Publisher("/agender_ros/people_descriptors", PeopleDescriptors, queue_size=1)
    self.database = self.initialize_database()
    self.init_nn()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.step(cv_image)
    except CvBridgeError as e:
      print(e)

  def initialize_database(self):
    """
    Reads the PNG images from ./people folder and creates a list of peoples
    The names of the image files are considered as their real names.
    For example;
    /people
      - mario.png
      - jennifer.png
      - melanie.png
    Returns:
    (tuple) (people_list, name_list) (features of people, names of people)
    """
    filenames = glob.glob(self.people_path + '*.png')
    people_list = []
    name_list = []
    for f in filenames:
      im = cv.imread(f, 1)
      im = im.astype(np.uint8)
      people_list.append(fr.face_encodings(im)[0])
      name_list.append(f.split('/')[-1].split('.')[0])
    return (people_list, name_list)

  def init_nn(self):
    # Desired width and height to process video.
    # Typically it should be smaller than original video frame
    # as smaller size significantly speeds up processing almost without affecting quality.
    self.width = 480
    self.height = 340
    self.resize_width_rate = 0.0
    self.resize_height_rate = 0.0

    # Choose which face detector to use. Select 'haar' or 'net'
    self.face_detector_kind = 'haar'

    # Choose what age and gender model to use. Specify 'ssrnet' or 'net'
    self.age_gender_kind = 'ssrnet'

    # hyper-parameters for bounding boxes shape
    # loading models
    self.emotion_model = load_model(self.models_path + 'emotion_recognition/_mini_XCEPTION.102-0.66.hdf5', compile=False)
    self.emotion_model._make_predict_function()
    self.emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]


    # Diagonal and line thickness are computed at run-time
    self.diagonal, self.line_thickness = None, None


    # Initialize face detector
    if (self.face_detector_kind == 'haar'):
        self.face_cascade = cv.CascadeClassifier(self.models_path + 'face_haar/haarcascade_frontalface_alt.xml')
    else:
        self.face_net = cv.dnn.readNetFromTensorflow(self.models_path + 'face_net/opencv_face_detector_uint8.pb', self.models_path + 'face_net/opencv_face_detector.pbtxt')

    self.gender_net = None
    self.age_net = None

    # Load age and gender models
    if (self.age_gender_kind == 'ssrnet'):
        # Setup global parameters
        self.face_size = 64
        self.face_padding_ratio = 0.10
        # Default parameters for SSR-Net
        stage_num = [3, 3, 3]
        lambda_local = 1
        lambda_d = 1
        # Initialize gender net
        self.gender_net = SSR_net_general(self.face_size, stage_num, lambda_local, lambda_d)()
        self.gender_net.load_weights(self.models_path + 'age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')
        self.gender_net._make_predict_function()
        # Initialize age net
        self.age_net = SSR_net(self.face_size, stage_num, lambda_local, lambda_d)()
        self.age_net.load_weights(self.models_path + 'age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')
        self.age_net._make_predict_function()

    else:
        # Setup global parameters
        self.face_size = 227
        self.face_padding_ratio = 0.0
        # Initialize gender detector
        self.gender_net = cv.dnn.readNetFromCaffe(self.models_path + 'age_gender_net/deploy_gender.prototxt', self.models_path + 'age_gender_net/gender_net.caffemodel')
        # Initialize age detector
        self.age_net = cv.dnn.readNetFromCaffe(self.models_path + 'age_gender_net/deploy_age.prototxt', self.models_path + 'age_gender_net/age_net.caffemodel')
        # Mean values for gender_net and age_net
        self.Genders = ['Male', 'Female']
        self.Ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

  def calculateParameters(self, height_orig, width_orig):
    area = self.width * self.height
    self.width = int(math.sqrt(area * width_orig / height_orig))
    self.height = int(math.sqrt(area * height_orig / width_orig))
    # Calculate diagonal
    self.diagonal = math.sqrt(self.height * self.height + self.width * self.width)
    # Calculate line thickness to draw boxes
    self.line_thickness = max(1, int(self.diagonal / 150))
    # Initialize output video writer
    #fps = cap.get(cv.CAP_PROP_FPS)
    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    #self.out = cv.VideoWriter('video.avi', fourcc=fourcc, fps=fps, frameSize=(self.width, self.height))

  def findFaces(self, img, confidence_threshold=0.7):
    # Get original width and height
    height = img.shape[0]
    width = img.shape[1]

    face_boxes = []

    if (self.face_detector_kind == 'haar'):
        # Get grayscale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect faces
        detections = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in detections:
            padding_h = int(math.floor(0.5 + h * self.face_padding_ratio))
            padding_w = int(math.floor(0.5 + w * self.face_padding_ratio))
            x1, y1 = max(0, x - padding_w), max(0, y - padding_h)
            x2, y2 = min(x + w + padding_w, width - 1), min(y + h + padding_h, height - 1)
            face_boxes.append([x1, y1, x2, y2])
    else:
        # Convert input image to 3x300x300, as NN model expects only 300x300 RGB images
        blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), mean=(104, 117, 123), swapRB=True, crop=False)
        # Pass blob through model and get detected faces
        face_net.setInput(blob)
        detections = face_net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence < confidence_threshold):
                continue
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            padding_h = int(math.floor(0.5 + (y2 - y1) * self.face_padding_ratio))
            padding_w = int(math.floor(0.5 + (x2 - x1) * self.face_padding_ratio))
            x1, y1 = max(0, x1 - padding_w), max(0, y1 - padding_h)
            x2, y2 = min(x2 + padding_w, width - 1), min(y2 + padding_h, height - 1)
            face_boxes.append([x1, y1, x2, y2])

    return face_boxes
  def collectFaces(self, frame, face_boxes):
    faces = []
    # Process faces
    for i, box in enumerate(face_boxes):
      # Convert box coordinates from resized frame_bgr back to original frame
      box_orig = [
        int(round(box[0] * self.width_orig / self.width)),
        int(round(box[1] * self.height_orig / self.height)),
        int(round(box[2] * self.width_orig / self.width)),
        int(round(box[3] * self.height_orig / self.height)),
      ]
      # Extract face box from original frame
      face_bgr = frame[
        max(0, box_orig[1]):min(box_orig[3] + 1, self.height_orig - 1),
        max(0, box_orig[0]):min(box_orig[2] + 1, self.width_orig - 1),
        :
      ]
      faces.append(face_bgr)
    return faces

  def predictEmotion(self,face):
    roi = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    roi = cv.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = self.emotion_model.predict(roi)[0]
    emotion_probability = np.max(preds)
    return (self.emotions[preds.argmax()], preds.argmax(), emotion_probability)

  def recognizer(self,face):
    face_locations = fr.face_locations(face)
    face_features = fr.face_encodings(face, face_locations)
    label = "Unknown"
    for features, (top, right, bottom, left) in zip(face_features, face_locations):
      matches = fr.compare_faces(self.database[0], features)
      if True in matches:
        ind = matches.index(True)
        label = self.database[1][ind]
    return label
  def descriptors2Msg(self, gender, age, emotion, recognition, face_box):
    descriptor_msg = PersonDescriptors()
    if gender < 0.5:
      gender = PersonDescriptors.FEMALE
    else:
      gender = PersonDescriptors.MALE

    if math.isnan(age):
      age = 0
    descriptor_msg.gender = gender
    descriptor_msg.age = int(age)
    descriptor_msg.emotion = emotion
    descriptor_msg.name = recognition
    descriptor_msg.face_xmin = int(face_box[0] * self.resize_width_rate)
    descriptor_msg.face_ymin = int(face_box[1] * self.resize_height_rate)
    descriptor_msg.face_xmax = int(face_box[2] * self.resize_width_rate)
    descriptor_msg.face_ymax = int(face_box[3] * self.resize_height_rate)

    return descriptor_msg

  def predictAgeGenderEmotion(self, frame, face_boxes):
    labels = []
    descriptor_list = []
    # Collect all faces into matrix
    recognition = "Unknown"
    faces = self.collectFaces(frame, face_boxes)
    if (self.age_gender_kind == 'ssrnet'):
      # Convert faces to N,64,64,3 blob
      blob = np.empty((len(faces), self.face_size, self.face_size, 3))
      for i, face_bgr in enumerate(faces):
        blob[i, :, :, :] = cv.resize(face_bgr, (64, 64))
        blob[i, :, :, :] = cv.normalize(blob[i, :, :, :], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # Predict gender and age
        genders = self.gender_net.predict(blob)
        ages = self.age_net.predict(blob)

        emotion, emotion_index, accuracy = self.predictEmotion(face_bgr)
        #recognition = self.recognizer(face_bgr)
        if (accuracy < 0.4):
            descriptor = self.descriptors2Msg(genders[0], ages[0], 0, recognition, face_boxes[i])
        else:
            descriptor = self.descriptors2Msg(genders[0], ages[0], emotion_index + 1, recognition, face_boxes[i])
        descriptor_list.append(descriptor)
        #  Construct labels
        labels = ['{},{},{},{}'.format(
            recognition,
            'Male' if (gender >= 0.5) else 'Female',
            int(age) if not math.isnan(age) else int(0),
            emotion)
          for (gender, age) in zip(genders, ages)]
        #  Construct labels
        #labels = ['{},{}'.format('Male' if (gender >= 0.5) else 'Female', int(age)) for (gender, age) in zip(genders, ages)]
    else:
      # Convert faces to N,3,227,227 blob
      blob = cv.dnn.blobFromImages(faces, scalefactor=1.0, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
      # Predict gender
      self.gender_net.setInput(blob)
      genders = self.gender_net.forward()
      # Predict age
      self.age_net.setInput(blob)
      ages = self.age_net.forward()
      emotion, emotion_index = self.predictEmotion(face_bgr);
      #recognition = self.recognizer(faces_bgr)
      # descriptor = self.descriptors2Msg(genders[0], ages[0], emotion_index, recognition, face_boxes[i])
      # descriptor_list.append(descriptor)
      #  Construct labels
      labels = ['{},{},{},{}'.format(recognition, self.Genders[gender.argmax()], self.Ages[age.argmax()], emotion) for (gender, age) in zip(genders, ages)]
    return labels, descriptor_list

  def step(self, frame):
    # Calculate parameters if not yet
    raw_height, raw_width, raw_channels = frame.shape
    self.resize_height_rate = float(raw_height) / float(self.height)
    self.resize_width_rate = float(raw_width) / float(self.width)
    if (self.diagonal is None):
      self.height_orig, self.width_orig = frame.shape[0:2]
      self.calculateParameters(self.height_orig, self.width_orig)
    # Resize, Convert BGR to HSV
    if ((self.height, self.width) != frame.shape[0:2]):
      frame_bgr = cv.resize(frame, dsize=(self.width, self.height), fx=0, fy=0)
    else:
      frame_bgr = frame
    # Detect faces
    face_boxes = self.findFaces(frame_bgr)
    # Make a copy of original image
    faces_bgr = frame_bgr.copy()

    if (len(face_boxes) > 0):
      # Get age and gender
      labels, descriptors = self.predictAgeGenderEmotion(frame, face_boxes)
      for desc in descriptors:
          color = []
          if ((desc.emotion == 4) or (desc.emotion == 6)):
            color = (0, 255, 0)
          elif (desc.emotion == 0 or desc.emotion == 7):
            color = (128, 128, 128)
          else:
            color = (0, 0, 255)

          cv.rectangle(
           faces_bgr,
           (int(desc.face_xmin / self.resize_width_rate), int(desc.face_ymin / self.resize_height_rate)),
           (int(desc.face_xmax / self.resize_width_rate), int(desc.face_ymax / self.resize_height_rate)),
           color,
           thickness=self.line_thickness,
           lineType=8)
      # Draw boxes in faces_bgr image
      #for (x1, y1, x2, y2) in face_boxes:
     #    cv.rectangle(faces_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=self.line_thickness, lineType=8)

      msg = PeopleDescriptors()
      msg.people_descriptors = descriptors
      self.descriptors_pub.publish(msg)
      # Draw labels and build msg
      for (label, box) in zip(labels, face_boxes):
        cv.putText(faces_bgr, label, org=(box[0], box[1] - 10), fontFace=cv.FONT_HERSHEY_PLAIN,
                           fontScale=1, color=(0, 64, 255), thickness=1, lineType=cv.LINE_AA)
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(faces_bgr, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ag = agender_ros()
  rospy.init_node('agender_ros_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
