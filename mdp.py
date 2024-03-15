import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

camera_url = 'rtsp://admin:ict@kmitl@192.168.8.111:554/Streaming/Channels/101'
cap = cv2.VideoCapture(camera_url)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 255, 5), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(225, 255, 0), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Camera Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

mp_drawing.DrawingSpec

def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return int(angle)

        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#
        image_height, image_width, _ = image.shape

        #coordinate_NOSE
        x_coodinate_NOSE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
        y_coodinate_NOSE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height

    #coordinate_LEFT_SHOULDER
        x_coodinate_LEFT_SHOULDER = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
        y_coodinate_LEFT_SHOULDER = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

        coodinate_LEFT_SHOULDER = [x_coodinate_LEFT_SHOULDER,y_coodinate_LEFT_SHOULDER]

        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        print(coodinate_LEFT_SHOULDER)


        print(image.shape)
        print([image_height,image_width])
