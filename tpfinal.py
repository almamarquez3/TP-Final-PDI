import cv2
import numpy as np

# Cargamos el detector preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abrimos la cámara (0) o un video
cap = cv2.VideoCapture('video 10.mp4')  # poné "video.mp4" si querés abrir un archivo
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertimos a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectamos las caras
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Dibujamos los rectángulos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostramos el resultado
    cv2.imshow('Detección de caras', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
