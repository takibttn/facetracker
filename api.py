import sys
import cv2
import numpy as np
import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore
import base64
from mtcnn import MTCNN
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                                 QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
                                 QComboBox, QSlider, QCheckBox, QMessageBox, QInputDialog,
                                 QGroupBox, QTabWidget, QFrame, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPalette


# ==================== Firebase Setup ====================
def initialize_firebase():
    """Initialize Firebase connection"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
    except ValueError:
        # Use your own Firebase credentials file path here
        cred_path = os.path.join(os.path.dirname(__file__), "firebase_credentials.json")
        # If credentials file doesn't exist, inform the user
        if not os.path.exists(cred_path):
            print(f"ERROR: Firebase credentials file not found at {cred_path}")
            print("Please download your Firebase service account credentials JSON file")
            print("and save it as 'firebase_credentials.json' in the same directory as this script.")
            return None

        # Initialize Firebase with credentials
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    # Return Firestore database client
    return firestore.client()


# ==================== Optimized Body Detector ====================
class OptimizedBodyDetector:
    def __init__(self):
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Optimized parameters
        self.win_stride = (8, 8)
        self.padding = (16, 16)
        self.scale = 1.05
        self.hit_threshold = 0.0

        # Tracking variables
        self.prev_boxes = []
        self.min_confidence = 0.5

        # Frame processing
        self.frame_skip = 4
        self.frame_count = 0

    def smart_detect(self, frame):
        """Robust detection with version compatibility"""
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return self.prev_boxes

        # Process at lower resolution
        height, width = frame.shape[:2]
        scale_factor = 0.75
        small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Universal detection handling
        try:
            # Try newer OpenCV return format
            found, weights = self.hog.detectMultiScale(
                gray,
                winStride=self.win_stride,
                padding=self.padding,
                scale=self.scale
            )
            found = [(x, y, w, h) for (x, y, w, h), weight in zip(found, weights)]
        except:
            # Fallback to older OpenCV return format
            found = self.hog.detectMultiScale(
                gray,
                winStride=self.win_stride,
                padding=self.padding,
                scale=self.scale
            )[0]

        # Scale back to original size
        current_boxes = []
        for (x, y, w, h) in found:
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            confidence = 1.0  # Default confidence if not available
            current_boxes.append((x, y, x+w, y+h, confidence))

        # Simple tracking between frames
        if self.prev_boxes:
            tracked_boxes = []
            for curr in current_boxes:
                matched = False
                for prev in self.prev_boxes:
                    if self._boxes_overlap(curr, prev):
                        # Average positions for smoothness
                        x1 = int(0.3*curr[0] + 0.7*prev[0])
                        y1 = int(0.3*curr[1] + 0.7*prev[1])
                        x2 = int(0.3*curr[2] + 0.7*prev[2])
                        y2 = int(0.3*curr[3] + 0.7*prev[3])
                        conf = max(curr[4], prev[4])
                        tracked_boxes.append((x1, y1, x2, y2, conf))
                        matched = True
                        break
                if not matched:
                    tracked_boxes.append(curr)
            self.prev_boxes = tracked_boxes
        else:
            self.prev_boxes = current_boxes

        return self.prev_boxes

    def _boxes_overlap(self, box1, box2):
        """Check if two boxes overlap significantly"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        return intersection_area > 0.3 * box1_area  # 30% overlap


# ==================== Centroid Tracker ====================
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        rects_arr = np.array(rects)
        input_centroids = np.column_stack((
            (rects_arr[:, 0] + rects_arr[:, 2]) // 2,
            (rects_arr[:, 1] + rects_arr[:, 3]) // 2
        ))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = np.array(list(self.objects.keys()))
            object_centroids = np.array(list(self.objects.values()))

            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in set(range(D.shape[1])) - used_cols:
                self.register(input_centroids[col])

        return self.objects


# ==================== MTCNN Face System with OpenCV DNN face recognition ====================
class MTCNNFaceSystem:
    def __init__(self, db, collection_name="face_embeddings", recognition_threshold=0.6):
        self.db = db  # Firestore database reference
        self.collection_name = collection_name
        self.recognition_threshold = recognition_threshold
        self.known_face_embeddings = []
        self.known_face_names = []
        self.known_face_ids = []

        # Initialize MTCNN for face detection
        self.detector = MTCNN()

        # Initialize OpenCV DNN face recognition model (OpenFace)
        model_file = os.path.join(os.path.dirname(__file__), "openface_nn4.small2.v1.t7")
        if not os.path.exists(model_file):
            print(f"WARNING: OpenFace model not found at {model_file}")
            print("Downloading it would be ideal, but for now we'll use a simpler embedding method")
            self.face_net = None
        else:
            self.face_net = cv2.dnn.readNetFromTorch(model_file)

        # Standard face image size for embeddings
        self.face_img_size = (96, 96)

        # Load existing embeddings from Firestore
        self.load_embeddings_from_firestore()

    def load_embeddings_from_firestore(self):
        """Load face embeddings from Firestore database"""
        try:
            if self.db is None:
                print("Firebase not initialized. Running in local-only mode.")
                return

            # Clear existing data
            self.known_face_embeddings = []
            self.known_face_names = []
            self.known_face_ids = []

            # Get all documents from the collection
            face_docs = self.db.collection(self.collection_name).stream()

            for doc in face_docs:
                doc_data = doc.to_dict()
                # Convert the embedding string to numpy array
                embedding_str = doc_data.get('embedding')
                if embedding_str:
                    # Convert from base64 string to numpy array
                    embedding_bytes = base64.b64decode(embedding_str)
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                    self.known_face_embeddings.append(embedding)
                    self.known_face_names.append(doc_data.get('name', 'Unknown'))
                    self.known_face_ids.append(doc.id)

            print(f"Loaded {len(self.known_face_names)} faces from Firestore")

        except Exception as e:
            print(f"Error loading embeddings from Firestore: {e}")

    def detect_faces(self, frame):
        try:
            # Convert from BGR (OpenCV) to RGB (MTCNN expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            detections = self.detector.detect_faces(rgb_frame)

            rects = []
            for detection in detections:
                if detection['confidence'] > 0.9:  # Filter by confidence
                    x, y, width, height = detection['box']
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)

                    if x2 > x1 and y2 > y1:
                        rects.append([x1, y1, x2, y2])

            return rects
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def get_face_embedding(self, face_img):
        if self.face_net is not None:
            # Using OpenCV DNN with OpenFace model
            try:
                # Resize to network's expected input
                face_blob = cv2.dnn.blobFromImage(face_img, 1.0/255, self.face_img_size, (0, 0, 0), swapRB=True, crop=False)

                # Set the input and get the embedding output
                self.face_net.setInput(face_blob)
                embedding = self.face_net.forward()

                # Return flattened and normalized embedding
                return embedding.flatten()
            except Exception as e:
                print(f"OpenFace embedding error: {e}")
                return self._get_simple_embedding(face_img)
        else:
            # Fallback to simple embedding
            return self._get_simple_embedding(face_img)

    def _get_simple_embedding(self, face_img):
        """Create a simple face embedding using HOG features and LBP"""
        try:
            # Resize image
            resized = cv2.resize(face_img, self.face_img_size)

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Calculate HOG (Histogram of Oriented Gradients)
            # Parameters for HOG
            win_size = (96, 96)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9

            # Create HOG descriptor
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            hog_features = hog.compute(gray)

            # Calculate LBP (Local Binary Pattern) histogram
            # Simple LBP implementation
            radius = 1
            n_points = 8 * radius
            lbp = np.zeros_like(gray)
            for i in range(radius, gray.shape[0] - radius):
                for j in range(radius, gray.shape[1] - radius):
                    center = gray[i, j]
                    binary = []
                    for k in range(n_points):
                        # Sample points around the center pixel
                        theta = 2 * np.pi * k / n_points
                        x = j + radius * np.cos(theta)
                        y = i - radius * np.sin(theta)
                        # Bilinear interpolation
                        x1, y1 = int(x), int(y)
                        x2, y2 = min(x1 + 1, gray.shape[1] - 1), min(y1 + 1, gray.shape[0] - 1)
                        dx, dy = x - x1, y - y1
                        value = (1 - dx) * (1 - dy) * gray[y1, x1] + \
                                dx * (1 - dy) * gray[y1, x2] + \
                                (1 - dx) * dy * gray[y2, x1] + \
                                dx * dy * gray[y2, x2]
                        binary.append(1 if value >= center else 0)
                    # Convert binary to decimal
                    binary_pattern = 0
                    for k, val in enumerate(binary):
                        binary_pattern += val * (2 ** k)
                    lbp[i, j] = binary_pattern

            # Calculate LBP histogram
            hist_lbp = cv2.calcHist([lbp.astype(np.float32)], [0], None, [256], [0, 256]).flatten()

            # Normalize histograms
            if np.sum(hist_lbp) != 0:
                hist_lbp = hist_lbp / np.sum(hist_lbp)

            # Combine features
            combined_features = np.concatenate([hog_features.flatten(), hist_lbp])

            # Normalize and return
            return combined_features.astype(np.float32)
        except Exception as e:
            print(f"Simple embedding error: {e}")
            return None

    def recognize_face(self, face_img):
        # Fix: Always return "Unknown" if there are no known faces
        if not self.known_face_embeddings:
            return "Unknown", float('inf')

        embedding = self.get_face_embedding(face_img)
        if embedding is None:
            return "Unknown", float('inf')

        # Initialize variables to track the best match
        min_distance = float('inf')
        best_match_name = "Unknown"
        best_match_distance = float('inf')


        for i, known_embedding in enumerate(self.known_face_embeddings):
            # Ensure embeddings have the same dimensions
            min_len = min(len(embedding), len(known_embedding))

            # Calculate cosine similarity
            dot_product = np.dot(embedding[:min_len], known_embedding[:min_len])
            norm_product = np.linalg.norm(embedding[:min_len]) * np.linalg.norm(known_embedding[:min_len])

            if norm_product == 0:
                similarity = 0
            else:
                similarity = dot_product / norm_product

            # Convert to distance (lower is better match)
            distance = 1.0 - similarity

            if distance < min_distance:
                min_distance = distance
                best_match_name = self.known_face_names[i]
                best_match_distance = distance

        # Fix: Properly handle recognition threshold
        # Lower threshold means more strict matching
        if min_distance <= self.recognition_threshold:
            return best_match_name, best_match_distance
        else:
            # Critical fix: Return "Unknown" if above threshold
            return "Unknown", best_match_distance

    def add_face(self, face_img, name):
        """Add a new face to Firestore database"""
        embedding = self.get_face_embedding(face_img)
        if embedding is None:
            print("Failed to generate face embedding")
            return False

        if self.db is None:
            print("Firebase not initialized. Cannot add face to database.")
            return False

        try:
            # Convert numpy array to bytes, then to base64 string for storage
            embedding_bytes = embedding.tobytes()
            embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')

            # Create a document in Firestore
            face_ref = self.db.collection(self.collection_name).document()
            face_ref.set({
                'name': name,
                'embedding': embedding_b64,
                'created_at': firestore.SERVER_TIMESTAMP
            })

            # Add to local cache
            self.known_face_embeddings.append(embedding)
            self.known_face_names.append(name)
            self.known_face_ids.append(face_ref.id)

            print(f"Added face: {name} (ID: {face_ref.id})")
            return True

        except Exception as e:
            print(f"Error adding face to Firestore: {e}")
            return False

    def add_multiple_faces(self, face_images, name):
        """Add multiple face images for the same person to improve accuracy"""
        success_count = 0
        total_images = len(face_images)

        if total_images == 0:
            print("No face images provided")
            return False

        for i, face_img in enumerate(face_images):
            print(f"Processing image {i+1}/{total_images} for {name}...")

            if self.add_face(face_img, name):
                success_count += 1

        print(f"Added {success_count}/{total_images} face images for {name}")
        return success_count > 0


# ==================== Video Thread ====================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    face_detected_signal = pyqtSignal(list)

    def __init__(self, face_system, body_detector):
        super().__init__()
        self.face_system = face_system
        self.body_detector = body_detector
        self._run_flag = True
        self.show_body_detection = True
        self.face_tracker = CentroidTracker(max_disappeared=30)
        self.body_tracker = CentroidTracker(max_disappeared=50)
        self.last_frame = None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


        self.face_frame_counter = 0  # Add this above the loop

        while self._run_flag:
            start_time = time.time()  # Start FPS timer

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            self.last_frame = frame.copy()
            display_frame = frame.copy()

            # === FACE DETECTION (every 3rd frame) ===
            self.face_frame_counter += 1
            if self.face_frame_counter % 3 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                small_face_rects = self.face_system.detect_faces(small_frame)
                self.last_face_rects = [
                    (int(x1 / 0.6), int(y1 / 0.6), int(x2 / 0.6), int(y2 / 0.6))
                    for (x1, y1, x2, y2) in small_face_rects
                ]
            face_rects = getattr(self, 'last_face_rects', [])
            face_objects = self.face_tracker.update(face_rects)

            detected_faces = []

            for (x1, y1, x2, y2) in face_rects:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                identity, confidence = self.face_system.recognize_face(face_img)
                detected_faces.append((identity, confidence))

                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{identity} ({confidence:.2f})"
                y_pos = max(y1 - 10, 10)
                cv2.putText(display_frame, label, (x1, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for object_id, centroid in face_objects.items():
                text_pos = (centroid[0] - 10, centroid[1] - 10)
                cv2.putText(display_frame, f"ID {object_id}", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(display_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # === BODY DETECTION ===
            if self.show_body_detection:
                body_boxes = self.body_detector.smart_detect(frame)
                body_rects = [[x1, y1, x2, y2] for (x1, y1, x2, y2, _) in body_boxes]
                body_objects = self.body_tracker.update(body_rects)

                for (x1, y1, x2, y2, conf) in body_boxes:
                    if conf > self.body_detector.min_confidence:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Body ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                for object_id, centroid in body_objects.items():
                    text_pos = (centroid[0] - 10, centroid[1] - 10)
                    cv2.putText(display_frame, f"B-{object_id}", text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.circle(display_frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

            # === HUD OVERLAY ===
            threshold_text = f"Recognition Threshold: {self.face_system.recognition_threshold:.2f}"
            cv2.putText(display_frame, threshold_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            faces_text = f"Known Faces: {len(self.face_system.known_face_names)}"
            cv2.putText(display_frame, faces_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            body_status = "ON" if self.show_body_detection else "OFF"
            body_text = f"Body Detection: {body_status}"
            cv2.putText(display_frame, body_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # === FPS Calculation ===
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # === Emit ===
            self.change_pixmap_signal.emit(display_frame)
            self.face_detected_signal.emit(detected_faces)

            self.msleep(15)  # Slightly faster delay to increase FPS

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# ==================== Main Window ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize Firebase
        self.db = initialize_firebase()
        if self.db is None:
            print("WARNING: Running without Firebase. Face data will not be saved.")

        # Initialize face system and body detector
        self.face_system = MTCNNFaceSystem(self.db, recognition_threshold=0.22)
        self.body_detector = OptimizedBodyDetector()

        # Set up the UI
        self.setWindowTitle("Face and Body Detection System")
        self.setGeometry(100, 100, 1200, 800)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: #ecf0f1;
                font-weight: bold;
            }
            QLabel {
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #7f8c8d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
            QCheckBox {
                color: #ecf0f1;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #7f8c8d;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
        """)

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)
        self.central_widget.setLayout(self.main_layout)

        # Left panel for video display
        self.video_panel = QVBoxLayout()
        self.video_panel.setSpacing(10)

        # Video display group
        video_group = QGroupBox("Live Camera Feed")
        video_group_layout = QVBoxLayout()
        video_group.setLayout(video_group_layout)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000;")
        video_group_layout.addWidget(self.video_label)

        # Control buttons
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(10)

        self.start_button = QPushButton("Start")
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.clicked.connect(self.start_capture)
        self.control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        self.control_layout.addWidget(self.stop_button)

        self.body_toggle = QCheckBox("Enable Body Detection")
        self.body_toggle.setChecked(True)
        self.body_toggle.stateChanged.connect(self.toggle_body_detection)
        self.control_layout.addWidget(self.body_toggle)

        video_group_layout.addLayout(self.control_layout)

        self.video_panel.addWidget(video_group)

        # Add video panel to main layout
        self.main_layout.addLayout(self.video_panel, stretch=2)

        # Right panel for controls and info
        self.control_panel = QVBoxLayout()
        self.control_panel.setSpacing(15)

        # Create tab widget for better organization
        self.tab_widget = QTabWidget()
        self.control_panel.addWidget(self.tab_widget)

        # Face Recognition Tab
        self.face_tab = QWidget()
        self.face_tab_layout = QVBoxLayout()
        self.face_tab_layout.setContentsMargins(10, 10, 10, 10)
        self.face_tab.setLayout(self.face_tab_layout)

        # Face recognition controls group
        face_control_group = QGroupBox("Face Recognition Settings")
        face_control_layout = QVBoxLayout()
        face_control_group.setLayout(face_control_layout)

        # Threshold control
        threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Recognition Threshold:")
        threshold_layout.addWidget(self.threshold_label)

        self.threshold_value = QLabel(f"{self.face_system.recognition_threshold:.2f}")
        self.threshold_value.setAlignment(Qt.AlignRight)
        threshold_layout.addWidget(self.threshold_value)

        face_control_layout.addLayout(threshold_layout)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.face_system.recognition_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        face_control_layout.addWidget(self.threshold_slider)

        # Add spacer
        face_control_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Face management buttons
        self.add_face_button = QPushButton("Add New Face")
        self.add_face_button.setIcon(QIcon.fromTheme("list-add"))
        self.add_face_button.clicked.connect(self.add_new_face)
        face_control_layout.addWidget(self.add_face_button)

        self.reload_button = QPushButton("Reload Face Data")
        self.reload_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.reload_button.clicked.connect(self.reload_face_data)
        face_control_layout.addWidget(self.reload_button)

        self.face_tab_layout.addWidget(face_control_group)

        # Add spacer to push everything up
        self.face_tab_layout.addStretch()

        # Add face tab to tab widget
        self.tab_widget.addTab(self.face_tab, "Face Recognition")

        # Detection Info Tab
        self.info_tab = QWidget()
        self.info_tab_layout = QVBoxLayout()
        self.info_tab_layout.setContentsMargins(10, 10, 10, 10)
        self.info_tab.setLayout(self.info_tab_layout)

        # Detection info group
        info_group = QGroupBox("Detection Information")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        self.detection_info_text = QTextEdit()
        self.detection_info_text.setReadOnly(True)
        info_layout.addWidget(self.detection_info_text)

        self.info_tab_layout.addWidget(info_group)

        # Add info tab to tab widget
        self.tab_widget.addTab(self.info_tab, "Detection Info")

        # Add control panel to main layout
        self.main_layout.addLayout(self.control_panel, stretch=1)

        # Initialize video thread
        self.video_thread = None

    def start_capture(self):
        """Start video capture and processing thread"""
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(self.face_system, self.body_detector)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.face_detected_signal.connect(self.update_detection_info)
            self.video_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_capture(self):
        """Stop video capture and processing thread"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: #000000;")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_video_frame(self, frame):
        """Update the video frame displayed in the GUI"""
        qt_img = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert OpenCV frame to QImage for display"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_detection_info(self, faces):
        """Update the detection information text box"""
        info_text = "Detected Faces:\n"
        if faces:
            for identity, confidence in faces:
                info_text += f"- {identity}: {confidence:.2f}\n"
        else:
            info_text += "No faces detected.\n"

        # Get body tracking info
        if self.video_thread and self.video_thread.show_body_detection:
            body_objects = self.video_thread.body_tracker.objects
            info_text += "\nDetected Bodies:\n"
            if body_objects:
                for id, centroid in body_objects.items():
                    info_text += f"- Body ID {id}: Center ({centroid[0]}, {centroid[1]})\n"
            else:
                info_text += "No bodies detected.\n"
        else:
            info_text += "\nBody detection is OFF.\n"

        self.detection_info_text.setText(info_text)

    def add_new_face(self):
        """Open a dialog to get the name of the new face and capture a frame"""
        if self.video_thread and self.video_thread.isRunning() and self.video_thread.last_frame is not None:
            name, ok = QInputDialog.getText(self, "Add New Face", "Enter name:")
            if ok and name:
                face_rects = self.face_system.detect_faces(self.video_thread.last_frame)
                if face_rects:
                    if len(face_rects) == 1:
                        x1, y1, x2, y2 = face_rects[0]
                        face_img = self.video_thread.last_frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            success = self.face_system.add_face(face_img, name)
                            if success:
                                QMessageBox.information(self, "Success", f"Face '{name}' added.")
                            else:
                                QMessageBox.critical(self, "Error", f"Failed to add face '{name}'.")
                        else:
                            QMessageBox.warning(self, "Warning", "Could not capture a valid face image.")
                    else:
                        QMessageBox.warning(self, "Warning", "Multiple faces detected. Please ensure only one face is visible.")
                else:
                    QMessageBox.warning(self, "Warning", "No face detected in the current frame.")
        else:
            QMessageBox.warning(self, "Warning", "Camera is not running or no frame captured yet.")

    def reload_face_data(self):
        """Reload face embeddings from Firestore"""
        self.face_system.load_embeddings_from_firestore()
        QMessageBox.information(self, "Info", "Face data reloaded from database.")

    def update_threshold(self, value):
        """Update the face recognition threshold based on slider value"""
        self.face_system.recognition_threshold = value / 100.0
        self.threshold_value.setText(f"{self.face_system.recognition_threshold:.2f}")

    def toggle_body_detection(self, state):
        """Toggle body detection on or off"""
        if self.video_thread:
            self.video_thread.show_body_detection = (state == Qt.Checked)

    def closeEvent(self, event):
        """Handle window closing event"""
        self.stop_capture()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())