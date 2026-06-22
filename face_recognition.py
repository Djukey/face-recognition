import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
facenet = FaceNet()


def get_face_embedding(image_path):
    """Load an image, detect the face, return the FaceNet embedding."""
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def get_face_embedding(image_path):
    """Load an image, detect the face, return the FaceNet embedding."""
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    max_size = 800
    h_orig, w_orig = image_rgb.shape[:2]
    if max(h_orig, w_orig) > max_size:
        scale = max_size / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))

    faces = detector.detect_faces(image_rgb)
    faces = detector.detect_faces(image_rgb)

    if len(faces) == 0:
        print("  WARNING: No face found in", image_path)
        return None

    x, y, w, h = faces[0]['box']
    face_crop = image_rgb[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (160, 160))
    face_batch = np.expand_dims(face_resized, axis=0)

    embedding = facenet.embeddings(face_batch)
    return embedding[0]


print("Building authorized database from database/ folder...\n")

database_folder = "database"
database_embeddings = []

for filename in os.listdir(database_folder):
    image_path = os.path.join(database_folder, filename)
    print("Processing:", filename)
    embedding = get_face_embedding(image_path)
    if embedding is not None:
        database_embeddings.append(embedding)

print("\nTotal embeddings collected:", len(database_embeddings))

authorized_embedding = np.mean(database_embeddings, axis=0)

print("Averaged authorized embedding shape:", authorized_embedding.shape)
print("First 5 numbers:", authorized_embedding[:5])

print("\n" + "=" * 60)
print("Testing all images in test_images/ folder")
print("=" * 60)

threshold = 1.0
test_folder = "test_images"

for filename in os.listdir(test_folder):
    if filename.lower().endswith(".webp"):
        print("\nSkipping", filename, "(.webp not supported reliably)")
        continue

    test_path = os.path.join(test_folder, filename)
    print("\nTest image:", filename)

    test_embedding = get_face_embedding(test_path)

    if test_embedding is None:
        print("  Could not detect a face. Skipping.")
        continue

    distance = np.linalg.norm(authorized_embedding - test_embedding)
    decision = "AUTHORIZED" if distance < threshold else "UNAUTHORIZED"

    print("  Distance:", round(distance, 4))
    print("  Result:  ", decision)

print("\n" + "=" * 60)
print("Starting real-time webcam recognition")
print("Press 'q' in the video window to quit")
print("=" * 60)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

while True:
    success, frame = camera.read()
    if not success:
        print("Could not read frame from webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        face_crop = frame_rgb[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (160, 160))
        face_batch = np.expand_dims(face_resized, axis=0)
        embedding = facenet.embeddings(face_batch)[0]

        distance = np.linalg.norm(authorized_embedding - embedding)

        if distance < threshold:
            label = "AUTHORIZED ({:.2f})".format(distance)
            color = (0, 255, 0)
        else:
            label = "UNAUTHORIZED ({:.2f})".format(distance)
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Live Face Recognition (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Webcam closed. Program finished.")