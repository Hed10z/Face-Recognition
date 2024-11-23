import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

# Function to get face embeddings
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    # Use the first detected face
    face = faces[0]
    landmarks = predictor(gray, face)
    embedding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))

    return embedding

# Load the reference image and extract its embedding
reference_image_path = "./img/ref.jpg"  # Replace with your image path
reference_image = cv2.imread(reference_image_path)
reference_embedding = get_face_embedding(reference_image)

if reference_embedding is None:
    print("No face detected in the reference image. Exiting.")
    exit()

# Start real-time webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face and compute embeddings for the live frame
    live_embedding = get_face_embedding(frame)

    if live_embedding is not None:
        # Compare embeddings using Euclidean distance
        similarity = distance.euclidean(reference_embedding, live_embedding)
        similarity_text = f"Similarity: {similarity:.2f}"

        # Define match threshold
        if similarity < 0.6:
            match_text = "MATCH"
            color = (0, 255, 0)  # Green
        else:
            match_text = "NO MATCH"
            color = (0, 0, 255)  # Red

        # Display similarity and match status
        cv2.putText(frame, similarity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, match_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the live video feed
    cv2.imshow("Face Comparison", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
