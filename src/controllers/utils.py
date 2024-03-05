import io
import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageDraw

watermark = cv2.imread("./assets/watermark.png")


def get_nparray_from_uploadfile(uploadfile: UploadFile) -> np.ndarray:
    contents = uploadfile.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image[:,:,:3]


def add_watermark(image: np.array, watermark: np.array = watermark) -> np.array:
    image = Image.fromarray(image)
    watermark = Image.fromarray(watermark)

    # Redimensionar imagen2 al tamaño de imagen1 si es necesario
    watermark = watermark.resize(image.size)

    # Superponer las imágenes
    imagen_final = Image.blend(image, watermark, alpha=0.25)
    
    # Save the watermarked image
    return np.array(imagen_final)


def round_img(image: Image):
    image = image.convert("RGBA")

    # Convert image to numpy array
    image_array = np.array(image)

    # Find bounding box
    min_x = np.min(np.where(image_array[:,:,3] > 0)[1])
    max_x = np.max(np.where(image_array[:,:,3] > 0)[1])
    min_y = np.min(np.where(image_array[:,:,3] > 0)[0])
    max_y = np.max(np.where(image_array[:,:,3] > 0)[0])

    # Compute the center and radius of the circumscribed circle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    radius = max(center_x, center_y)

    # Create a blank transparent image
    new_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(new_image)

    # Draw the circumscribed circumference with transparency outside the circle
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
              draw.point((x, y), fill=image.getpixel((x, y)))

    return new_image
import cv2
import numpy as np

def detect_face_in_image(img):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's Haar Cascade to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # In this example, just take the first detected face
    if len(faces) == 0:
        return None
    return faces[0]

def apply_vignette(image):
    bbox = detect_face_in_image(image)
    if bbox is None:
        # If no face is detected, return the original image
        return image
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)
    face_radius = max(w, h) * 0.85  # Reduced radius
    
    rows, cols, _ = image.shape
    vignette_mask = np.zeros((rows, cols), dtype=np.uint8)
    smooth_factor = 0.8  # Ultra smooth transition
    
    for i in range(rows):
        for j in range(cols):
            dist_to_center = np.sqrt((center[0]-j)**2 + (center[1]-i)**2)
            dist_to_edge = min(j, i, cols - j, rows - i)
            vignette_strength = (1.0 - dist_to_edge / max(rows, cols) * 2) * 255
            fade = np.clip((dist_to_center - face_radius) / (face_radius * smooth_factor), 0, 1)
            vignette_mask[i, j] = np.clip(vignette_strength * fade, 0, 255)
            
    vignette_image = cv2.merge([vignette_mask, vignette_mask, vignette_mask])
    
    # Create a mask to preserve the face
    face_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(face_mask, center, int(face_radius), (255, 255, 255), -1)
    face_image = cv2.bitwise_and(image, image, mask=face_mask)
    
    # Subtract face from the original image
    image_without_face = cv2.subtract(image, face_image)
    
    # Apply the vignette mask to the image without the face
    result_without_face = cv2.addWeighted(image_without_face, 1.0, vignette_image, -0.7, 0)
    
    # Combine the results to get the final image
    result = cv2.add(result_without_face, face_image)
    
    return result
