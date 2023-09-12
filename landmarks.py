import mediapipe as mp
import cv2
import numpy as np


class holostic_model():
    def __init__(self,min_detection_confidence=0.5,min_tracking_confidence=0.8):
        # calling mediapipe holistic model and assigning to valiables
        self.mp_holistic=mp.solutions.holistic # holistic model
        self.mp_drawing=mp.solutions.drawing_utils # utilities for holistic model to drawn landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        #self.holistic = self.mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
            
    def detection_model(self,img,model):
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable=False
        result=model.process(img)
        img.flags.writeable=True
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img,result
    
    def draw_style(self,img,result):
        #face landmark
        self.mp_drawing.draw_landmarks(img, result.face_landmarks,self.mp_holistic.FACEMESH_CONTOURS,
                                       self.mp_drawing.DrawingSpec(color=(220, 103, 112),thickness=1,circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(149, 234, 149),thickness=2,circle_radius=1))

        #pose landmark
        self.mp_drawing.draw_landmarks(img, result.pose_landmarks,self.mp_holistic.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(20, 19, 247),thickness=2,circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(149, 234, 149),thickness=2,circle_radius=1))
        #left hand landmark
        self.mp_drawing.draw_landmarks(img, result.left_hand_landmarks,self.mp_holistic.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(169, 42, 27),thickness=2,circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(254, 114, 236),thickness=2,circle_radius=1))
        #Right habd landmark
        self.mp_drawing.draw_landmarks(img, result.right_hand_landmarks,self.mp_holistic.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(254, 114, 236),thickness=2,circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(169, 42, 27),thickness=2,circle_radius=1))
        


    def display_video(self,source):
        cap=cv2.VideoCapture(source)

        #calling mediapipe line model and as to holistic variable
        with self.mp_holistic.Holistic(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as holistic:

        #capturing frames and looping to show as video
            while cap.isOpened():
                ret, frame=cap.read() # ret will be bool (when runing it will be true)
                # to strat while loop and below else is to break the loop
                if ret == True:
                    #Converting to rgb and unwriteable to better with mediapipe and reversing and output image and result of model
                    img,result=self.detection_model(frame,holistic)

                    #drawing the result in the image
                    self.draw_style(img,result)

                    # to mirror the output image parameter 1 is use to flip along the y axix.
                    image = cv2.flip(img,1)

                    # displaying the resluted image
                    cv2.imshow('Capturing',image)

                    if cv2.waitKey(25) & 0xff == ord("q"):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()


    def key_values(self,result):
        pose = np.array([[result.x,result.y,result.z] for result in result.pose_landmarks.landmark]) if result.pose_landmarks else np.zeros((33,3)) 
        face = np.array([[result.x,result.y,result.z] for result in result.face_landmarks.landmark]) if result.face_landmarks else np.zeros((468,3))
        right_hand = np.array([[result.x,result.y,result.y] for  result in result.right_hand_landmarks.landmark]) if result.right_hand_landmarks else np.zeros((21,3))
        left_hand = np.array([[result.x,result.y,result.z] for result in result.left_hand_landmarks.landmark]) if result.left_hand_landmarks else np.zeros((21,3))
        return np.concatenate([face,left_hand,pose,right_hand])
    
    
    def extract_values(self,source,display=False):
        values = []
        cap=cv2.VideoCapture(source)

        #calling mediapipe line model and as to holistic variable
        with self.mp_holistic.Holistic(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as holistic:

        #capturing frames and looping to show as video
            while cap.isOpened():
                ret, frame=cap.read() # ret will be bool (when runing it will be true)
                # to strat while loop and below else is to break the loop
                if ret == True:
                    #Converting to rgb and unwriteable to better with mediapipe and reversing and output image and result of model
                    img,result = self.detection_model(frame,holistic)
                    if display:
                        #drawing the result in the image
                        self.draw_style(img,result)
                        # to mirror the output image parameter 1 is use to flip along the y axix.
                        image = cv2.flip(img,1)
                        # displaying the resluted image
                        cv2.imshow('Capturing',image)

                    # to mirror the output image parameter 1 is use to flip along the y axix.
                    values.append(self.key_values(result))
                else:
                    break
            cap.release()
        return np.squeeze(values)
    
