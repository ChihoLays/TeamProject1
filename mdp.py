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

act_list = [' ', 'stand', 'walk', 'fall', 'lay down']
# demo
act = 1
activity = act_list[act]
print(activity)