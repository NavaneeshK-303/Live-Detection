import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from scipy.spatial import distance
import threading
import time
import csv
from datetime import datetime

# Constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
MAR_THRESHOLD = 0.75  # Mouth Aspect Ratio threshold
BLINK_RATE_THRESHOLD = 15  # Blinks per minute threshold
BLINK_WINDOW_SIZE = 10  # Window size for blink rate calculation
HEAD_TILT_THRESHOLD = 20  # Head tilt angle threshold
DROWSINESS_TIME_THRESHOLD = 5  # Time threshold for drowsiness (in seconds)

# Indices for landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]  # Left Eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # Right Eye
MOUTH = [61, 39, 0, 269, 291, 405, 17, 181, 406, 313, 14, 87, 178, 402, 318, 324, 308]  # Mouth

class DrowsinessDetector:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize MediaPipe Face Mesh and Pose
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp.solutions.pose.Pose()

        # Initialize OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # Variables
        self.blink_count = 0
        self.blink_times = []
        self.drowsiness_log = []
        self.eye_closed_start_time = None  # Track when eyes first close

    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)."""
        poi_A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        poi_B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        poi_C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (poi_A + poi_B) / (2 * poi_C)
        return ear

    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio (MAR)."""
        poi_A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        poi_B = distance.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        poi_C = distance.euclidean(mouth_landmarks[3], mouth_landmarks[5])
        poi_D = distance.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        mar = (poi_A + poi_B + poi_C) / (3 * poi_D)
        return mar

    def calculate_head_pose(self, face_landmarks, frame_shape):
        """Calculate head pose angles."""
        nose_tip = face_landmarks.landmark[4]
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]

        # Convert landmarks to pixel coordinates
        left_ear_px = (int(left_ear.x * frame_shape[1]), int(left_ear.y * frame_shape[0]))
        right_ear_px = (int(right_ear.x * frame_shape[1]), int(right_ear.y * frame_shape[0]))

        # Calculate head tilt angle
        dx = right_ear_px[0] - left_ear_px[0]
        dy = right_ear_px[1] - left_ear_px[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def speak_alert(self, message):
        """Speak alerts using text-to-speech."""
        self.engine.say(message)
        self.engine.runAndWait()

    def log_drowsiness(self, event):
        """Log drowsiness events to a CSV file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.drowsiness_log.append((timestamp, event))
        with open("drowsiness_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event])

    def process_frame(self, frame):
        """Process each frame to detect drowsiness."""
        # Convert frame to RGB (for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks
                left_eye_points = self.extract_landmarks(face_landmarks, LEFT_EYE, frame.shape)
                right_eye_points = self.extract_landmarks(face_landmarks, RIGHT_EYE, frame.shape)
                mouth_points = self.extract_landmarks(face_landmarks, MOUTH, frame.shape)

                # Compute metrics
                left_ear = self.calculate_ear(left_eye_points)
                right_ear = self.calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2
                mar = self.calculate_mar(mouth_points)
                head_angle = self.calculate_head_pose(face_landmarks, frame.shape)

                # Blink detection
                if avg_ear < EAR_THRESHOLD:
                    self.blink_count += 1
                    self.blink_times.append(time.time())
                    if len(self.blink_times) > BLINK_WINDOW_SIZE:
                        self.blink_times.pop(0)

                # Calculate blink rate (blinks per minute)
                blink_rate = self.calculate_blink_rate()

                # Detect drowsiness, yawning, head tilt, and low blink rate
                self.detect_drowsiness(frame, avg_ear, mar, head_angle, blink_rate)

                # Display metrics on frame
                self.display_metrics(frame, avg_ear, mar, blink_rate, head_angle)

        return frame

    def extract_landmarks(self, face_landmarks, indices, frame_shape):
        """Extract landmarks for given indices."""
        points = []
        for idx in indices:
            x = int(face_landmarks.landmark[idx].x * frame_shape[1])
            y = int(face_landmarks.landmark[idx].y * frame_shape[0])
            points.append((x, y))
        return points

    def calculate_blink_rate(self):
        """Calculate blink rate (blinks per minute)."""
        if len(self.blink_times) >= 2:
            return (len(self.blink_times) - 1) / (self.blink_times[-1] - self.blink_times[0]) * 60
        return 0

    def detect_drowsiness(self, frame, avg_ear, mar, head_angle, blink_rate):
        """Detect drowsiness, yawning, head tilt, and low blink rate."""
        # Drowsiness detection with 5-second delay
        if avg_ear < EAR_THRESHOLD:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = time.time()  # Start timer
            elif time.time() - self.eye_closed_start_time >= DROWSINESS_TIME_THRESHOLD:
                self.display_alert(frame, "DROWSINESS DETECTED!", (50, 400))  # Display alert at bottom-left
                threading.Thread(target=self.speak_alert, args=("Alert! Wake up!",)).start()
                self.log_drowsiness("Drowsiness Detected")
        else:
            self.eye_closed_start_time = None  # Reset timer if eyes are open

        # Yawning detection
        if mar > MAR_THRESHOLD:
            self.display_alert(frame, "YAWN DETECTED!", (50, 450))  # Display alert at bottom-left
            threading.Thread(target=self.speak_alert, args=("Alert! You are yawning.",)).start()
            self.log_drowsiness("Yawning Detected")

        # Head tilt detection
        if abs(head_angle) > HEAD_TILT_THRESHOLD:
            self.display_alert(frame, "HEAD TILT DETECTED!", (50, 500))  # Display alert at bottom-left
            threading.Thread(target=self.speak_alert, args=("Alert! Keep your head straight.",)).start()
            self.log_drowsiness("Head Tilt Detected")

        # Low blink rate detection
        if blink_rate < BLINK_RATE_THRESHOLD:
            self.display_alert(frame, "LOW BLINK RATE!", (50, 550))  # Display alert at bottom-left
            threading.Thread(target=self.speak_alert, args=("Alert! Blink more often.",)).start()
            self.log_drowsiness("Low Blink Rate")

    def display_alert(self, frame, message, position):
        """Display alert messages on the frame."""
        cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def display_metrics(self, frame, avg_ear, mar, blink_rate, head_angle):
        """Display metrics on the frame."""
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f} bpm", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Head Angle: {head_angle:.1f} deg", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def run(self):
        """Run the drowsiness detection system."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame
                frame = self.process_frame(frame)

                # Show output
                cv2.imshow("Drowsiness Detector", frame)

                # Check if the window is closed
                if cv2.getWindowProperty("Drowsiness Detector", cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.engine.stop()

# Main entry point
if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
