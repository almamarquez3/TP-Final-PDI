import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Config ------------------
VIDEO_PATH = r'C:\Users\almam\OneDrive\Escritorio\UNIV\4 CUARTO AÑO\SEGUNDO CUATRI\PDI\TP Final py\videos\Video 23.mp4'
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
        roi_face = frame[fy:fy+fh, fx:fx+fw]
        G_cara_completa= np.mean(roi_face[:, :, 1])  # único valor G(t)
        for nombre, frac in ROI_FRAC.items():
            x, y, w, h = frac_to_rect((fx, fy, fw, fh), frac, W, H)
            roi = frame[y:y+h, x:x+w]
        G_values = {}
        Gfrente=[]
        Gpomulo1=[]
        Gpomulo2=[]
        Gtabique=[]
        if roi.size > 0:
            G = roi[:, :, 1]  # canal verde

        # Guardar en la lista que corresponde
        if nombre == "frente":
            Gfrente.append(G)
        elif nombre == "tabique":
            Gtabique.append(G)
        elif nombre == "mejilla_d":
            Gpomulo1.append(G)
        elif nombre == "mejilla_i":
            Gpomulo2.append(G)

        # Guardar muestra
        serie_G.append(G_cara_completa)
        last_G = G_cara_completa

        # Dibujar ROIs (opcional)
        for nombre in ROI_FRAC:
            x, y, w, h = frac_to_rect((fx, fy, fw, fh), ROI_FRAC[nombre], W, H)

            if nombre == "frente":
                # Rectángulo verde clásico
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elif nombre == "tabique":
                # Rectángulo más pequeño y centrado (1/3 del tamaño)
                shrink = 0.33
                w2 = int(w * shrink)
                h2 = int(h * shrink)
                cx, cy = x + w // 2, y + h // 2
                x2, y2 = cx - w2 // 2, cy - h2 // 2
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

            elif nombre in ["mejilla_d", "mejilla_i"]:
                # Óvalo no muy grande centrado en la mejilla
                shrink = 0.6  # cuanto menor, más chico el óvalo
                w2 = int(w * shrink)
                h2 = int(h * shrink)
                cx, cy = x + w // 2, y + h // 2
                cv2.ellipse(
                    frame,
                    (cx, cy),
                    (w2 // 2, h2 // 2),
                    0,          # rotación
                    0, 360,     # ángulo inicial y final
                    (0, 255, 0),
                    2
                )

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
mucc = float(np.mean(serie_G))
sigmacc = float(np.std(serie_G, ddof=0))

mufr = float(np.mean(Gfrente))
sigfr = float(np.std(Gfrente, ddof=0))

mup1 = float(np.mean(Gpomulo1))
sigp1 = float(np.std(Gpomulo1, ddof=0))

mup2 = float(np.mean(Gpomulo2))
sigp2 = float(np.std(Gpomulo2, ddof=0))

muta = float(np.mean(Gtabique))
sigta = float(np.std(Gtabique, ddof=0))

zcc = (serie_G - mucc) / sigmacc
zfr = (Gfrente - mufr) / sigfr
zp1 = (Gpomulo1 - mup1) / sigp1
zp2 = (Gpomulo2 - mup2) / sigp2
zta = (Gtabique - muta) / sigta

t = np.arange(N) / fps


# ------------------ FFT ------------------
dt = 1.0 / fps   # seguro porque fps > 0
freqs = np.fft.rfftfreq(N, d=dt)
Zcc = np.abs(np.fft.rfft(zcc)) / N
i0=0
i1=0
for i in range(len(freqs)):
    if freqs[i]>=0.9 and freqs[i]<=1.1:
        i0=i
    if freqs[i]>=1.9 and freqs[i]<=2.1:
        i1=i  

Z_rangocc=Zcc[i0:i1+1]
maximocc=max(Z_rangocc)
for i in range(len(Zcc)):
    if Zcc[i]==maximocc:
        lpmcc=freqs[i]
print("Latido por minuto estimado:caracompleta ", lpmcc*60, "bpm")
print(maximocc)

Zfr = np.abs(np.fft.rfft(zfr)) / N
i0=0
i1=0
for i in range(len(freqs)):
    if freqs[i]>=0.9 and freqs[i]<=1.1:
        i0=i
    if freqs[i]>=1.9 and freqs[i]<=2.1:
        i1=i  

Z_rangofr=Zfr[i0:i1+1]
maximofr=max(Z_rangofr)
for i in range(len(Zfr)):
    if Zfr[i]==maximofr:
        lpmfr=freqs[i]
print("Latido por minuto estimado:frente ", lpmfr*60, "bpm")
print(maximofr)

Zp1 = np.abs(np.fft.rfft(zp1)) / N
i0=0
i1=0
for i in range(len(freqs)):
    if freqs[i]>=0.9 and freqs[i]<=1.1:
        i0=i
    if freqs[i]>=1.9 and freqs[i]<=2.1:
        i1=i  

Z_rangop1=Zp1[i0:i1+1]
maximop1=max(Z_rangop1)
for i in range(len(Zp1)):
    if Zp1[i]==maximop1:
        lpmp1=freqs[i]
print("Latido por minuto estimado:caracompleta ", lpmp1*60, "bpm")
print(maximop1)

Zp2 = np.abs(np.fft.rfft(zp2)) / N
i0=0
i1=0
for i in range(len(freqs)):
    if freqs[i]>=0.9 and freqs[i]<=1.1:
        i0=i
    if freqs[i]>=1.9 and freqs[i]<=2.1:
        i1=i  

Z_rangop2=Zp2[i0:i1+1]
maximop2=max(Z_rangop2)
for i in range(len(Zp2)):
    if Zp2[i]==maximop2:
        lpmp2=freqs[i]
print("Latido por minuto estimado:caracompleta ", lpmp2*60, "bpm")
print(maximop2)

Zta = np.abs(np.fft.rfft(zta)) / N
i0=0
i1=0
for i in range(len(freqs)):
    if freqs[i]>=0.9 and freqs[i]<=1.1:
        i0=i
    if freqs[i]>=1.9 and freqs[i]<=2.1:
        i1=i  

Z_rangota=Zta[i0:i1+1]
maximota=max(Z_rangota)
for i in range(len(Zta)):
    if Zta[i]==maximota:
        lpmt=freqs[i]
print("Latido por minuto estimado:tabique ", lpmt*60, "bpm")
print(maximota)

# ------------------ Gráficos (a) y (b) ------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# (a) Señal temporal estandarizada
ax1.plot(t, zcc, label='G')
ax1.set_xlabel('Segundos')
ax1.set_ylabel('Intensidad')
ax1.legend(loc='upper right')

# (b) FFT
ax2.plot(freqs, Zcc, label='FFT')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud')
ax2.set_xlim(0, 15)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
