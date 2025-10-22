import cv2
import numpy as np

def clamp_rect(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

FRONT_FRAC = dict(x=0.300, y=0.100, w=0.400, h=0.125)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

in_path  = "video 2.mp4"
out_path = "salida_con_frente.mp4"

cap = cv2.VideoCapture(in_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80)
    )

    if len(faces) > 0:
        i = np.argmax([w*h for (x,y,w,h) in faces])
        x, y, w, h = faces[i]

        fx = int(x + FRONT_FRAC["x"] * w)
        fy = int(y + FRONT_FRAC["y"] * h)
        fw = int(FRONT_FRAC["w"] * w)
        fh = int(FRONT_FRAC["h"] * h)
        fx, fy, fw, fh = clamp_rect(fx, fy, fw, fh, W, H)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)

        center = (x + w//2, y + h//2)
        axes   = (w//2, h//2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 2)

    writer.write(frame)
    cv2.imshow("Procesando...", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Guardado: {out_path}")
