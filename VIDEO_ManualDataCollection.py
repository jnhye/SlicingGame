from SetupTesting import *
from KPCreateFolders import *
import time
import cv2
import os
from tkinter import Tk, filedialog


DATA_PATH_vid = os.path.join('DATA_VIDEO')

cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if fps is 0

# Define codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            # Save video to: DATA_VIDEO/action/sequence/action_sequence.mp4
            video_filename = os.path.join(DATA_PATH_vid, action, str(sequence), f"{action}_{sequence}.mp4")
            out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                # draw_styled_landmarks(image, results)
                out.write(image)

                if frame_num == 0:
                    cv2.putText(image, f'STARTING COLLECTION - {action}', (120, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            out.release()

            # Rest break between recordings
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                remaining_time = 3 - int(time.time() - start_time)
                cv2.putText(frame, f'Rest {remaining_time}', (300, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Brief buffer to stabilize webcam
            buffer_start_time = time.time()
            while time.time() - buffer_start_time < 0.5:
                cap.read()

    cap.release()
    cv2.destroyAllWindows()
