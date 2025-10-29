import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Config ------------------
VIDEO_PATH = 0
USE_HOLD_LAST = True      # True: sostener último valor cuando no hay detección
USE_INTERPOLATE_NAN = True  # True: si hay NaN, interpolar al final

ROI_FRAC = {
    "frente":   dict(x=0.300, y=0.100, w=0.400, h=0.125),
    "tabique":  dict(x=0.420, y=0.346, w=0.160, h=0.104),
    "mejilla_d":dict(x=0.218, y=0.569, w=0.138, h=0.147),
    "mejilla_i":dict(x=0.644, y=0.569, w=0.138, h=0.147),
}

# ------------------ Utilidades ------------------
def clamp_rect(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def frac_to_rect(face_rect, frac, frame_W, frame_H):
    fx, fy, fw, fh = face_rect
    x = int(fx + frac["x"] * fw)
    y = int(fy + frac["y"] * fh)
    w = int(frac["w"] * fw)
    h = int(frac["h"] * fh)
    return clamp_rect(x, y, w, h, frame_W, frame_H)

# ------------------ Captura ------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(VIDEO_PATH)

# FPS robusto
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0
print(f"FPS detectado: {fps:.2f}")

# Serie temporal del G promedio total (frente + tabique + mejillas)
serie_G = []
last_G = None  # último valor válido para hold-last

# ------------------ Loop de frames ------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        (fx, fy, fw, fh) = max(faces, key=lambda r: r[2]*r[3])

        # Promedio G por ROI
        mean_G_values = {}
        for nombre, frac in ROI_FRAC.items():
            x, y, w, h = frac_to_rect((fx, fy, fw, fh), frac, W, H)
            roi = frame[y:y+h, x:x+w]
            G = roi[:, :, 1]
            mean_G_values[nombre] = float(np.mean(G))

        # Promedio total entre regiones
        regiones = ["frente", "tabique", "mejilla_d", "mejilla_i"]
        G_prom_total = np.mean([mean_G_values[r] for r in regiones])

        # Guardar muestra
        serie_G.append(G_prom_total)
        last_G = G_prom_total

        # Overlay informativo (opcional)
        cv2.putText(frame, f"G prom: {G_prom_total:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dibujar ROIs (opcional)
        for nombre in ROI_FRAC:
            x, y, w, h = frac_to_rect((fx, fy, fw, fh), ROI_FRAC[nombre], W, H)
            if nombre == "frente":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cx, cy = x + w//2, y + h//2
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    else:
        # Sin detección: sostener último valor o marcar NaN
        if USE_HOLD_LAST and (last_G is not None):
            serie_G.append(last_G)
        elif USE_INTERPOLATE_NAN:
            serie_G.append(np.nan)
        # si no querés agregar nada, comentá ambas y no se agrega muestra

    cv2.imshow("Promedio canal G", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------ Post-proceso ------------------
serie_G = np.asarray(serie_G, dtype=float)

# Interpolar si hay NaNs
if np.isnan(serie_G).any() and USE_INTERPOLATE_NAN:
    idx = np.arange(serie_G.size)
    valid = ~np.isnan(serie_G)
    if valid.sum() >= 2:
        serie_G = np.interp(idx, idx[valid], serie_G[valid])
    else:
        print("No hay suficientes datos válidos para interpolar/analizar.")
        raise SystemExit

N = len(serie_G)
if N < 8:
    print(f"Demasiado pocas muestras para FFT (N={N}).")
    raise SystemExit

# Estandarización Z-score
mu = float(np.mean(serie_G))
sigma = float(np.std(serie_G, ddof=0))
if not np.isfinite(sigma) or sigma == 0:
    print("Varianza no válida (sigma=0 o NaN). No se puede estandarizar.")
    raise SystemExit

Z = (serie_G - mu) / sigma
t = np.arange(N) / fps

# ------------------ FFT ------------------
dt = 1.0 / fps   # seguro porque fps > 0
freqs = np.fft.rfftfreq(N, d=dt)
fft_vals = np.abs(np.fft.rfft(Z)) / N

# ------------------ Gráficos (a) y (b) ------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# (a) Señal temporal estandarizada
ax1.plot(t, Z, label='G')
ax1.set_xlabel('Segundos')
ax1.set_ylabel('Intensidad')
ax1.legend(loc='upper right')

# (b) FFT
ax2.plot(freqs, fft_vals, label='FFT')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud')
ax2.set_xlim(0, 15)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
