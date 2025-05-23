from SetupTesting import *
from KPCreateFolders import *
import time


# Set parameters
VIDEO_DATA_PATH = os.path.join('DATA_VIDEO')
no_sequences = 30
sequence_length = 25


# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # NEW LOOP
    # Loop through actions
    for action in actions:
        action_path = os.path.join(VIDEO_DATA_PATH, action)
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder + no_sequences):
            video_path = os.path.join(action_path, str(sequence), f"{action}_{str(sequence)}.mp4")
            cap = cv2.VideoCapture(video_path)
            print("Processing :", video_path, " ", end="")

            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue


            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Mirror the frame
                frame = cv2.flip(frame, 1)

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                debugging = False
                if debugging:
                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    cv2.putText(image, 'Collecting frames for {} Video Number {}  Frame Number {}'.format(action, sequence, frame_num), (15, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(10)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
            print("DONE!")


    cap.release()
    cv2.destroyAllWindows()
