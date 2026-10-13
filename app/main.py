from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from deepface import DeepFace
from typing import List

import cv2
import numpy as np


app = FastAPI(
    title="Face Recognition API",
    version="1.0.0"
)


# Load face detector once when the application starts
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade")


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


def read_image(contents: bytes):
    image_array = np.frombuffer(contents, dtype=np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    return image


def detect_faces(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return faces

def crop_faces(image, faces):
    cropped_faces = []

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        cropped_faces.append(face)

    return cropped_faces

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = read_image(contents)

    faces = detect_faces(image)

    result = []

    for (x, y, w, h) in faces:
        result.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })

    return {
        "count": len(result),
        "faces": result
    }


@app.post("/detect/image")
async def detect_face_image(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = read_image(contents)
    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode image"
        )

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )

def crop_faces(image, faces):
    cropped_faces = []

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        cropped_faces.append(face)

    return cropped_faces

@app.post("/detect/crops")
async def detect_face_crops(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = read_image(contents)
    faces = detect_faces(image)

    if len(faces) == 0:
        raise HTTPException(
            status_code=404,
            detail="No face detected"
        )

    cropped_faces = crop_faces(image, faces)

    face = cropped_faces[0]

    success, encoded_image = cv2.imencode(
        ".jpg",
        face
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode cropped face"
        )

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )

@app.post("/embedding")
async def create_embedding(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = read_image(contents)

    embedding = create_face_embedding(image)

    return {
        "embedding_size": len(embedding),
        "embedding": embedding
    }

@app.post("/compare")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    contents1 = await file1.read()
    contents2 = await file2.read()

    image1 = read_image(contents1)
    image2 = read_image(contents2)

    embedding1 = create_face_embedding(image1)
    embedding2 = create_face_embedding(image2)

    similarity = cosine_similarity(
        embedding1,
        embedding2
    )

    return {
        "similarity": similarity
    }

def create_face_embedding(image):
    embedding_result = DeepFace.represent(
        img_path=image,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=True,
        align=True
    )

    if len(embedding_result) == 0:
        raise HTTPException(
            status_code=404,
            detail="No face detected"
        )

    embedding = embedding_result[0]["embedding"]

    return embedding

def cosine_similarity(
    embedding1: List[float],
    embedding2: List[float]
) -> float:
    vector1 = np.array(embedding1, dtype=np.float32)
    vector2 = np.array(embedding2, dtype=np.float32)

    if vector1.shape != vector2.shape:
        raise ValueError(
            "Embedding vectors must have the same size"
        )

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError(
            "Embedding vector cannot be zero"
        )

    similarity = np.dot(vector1, vector2) / (norm1 * norm2)

    return float(similarity)