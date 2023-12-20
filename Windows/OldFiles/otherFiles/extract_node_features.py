import cv2
import numpy as np
import scipy.signal
from skimage.filters import gabor_kernel

# Definisci i parametri per i filtri di Gabor
frequencies = [0.1, 0.3, 0.5]  # Frequenze dei filtri
orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Orientazioni dei filtri

# Dimensione della finestra intorno a ciascun landmark per l'applicazione di Gabor
window_size = 32


def extract_gabor_features(image, landmarks):
    features = []

    # Ciclo attraverso ciascun landmark
    for landmark in landmarks:
        x, y = landmarks[landmark]

        # Estrai la finestra intorno al landmark
        window = image[y - window_size // 2:y + window_size // 2,
                 x - window_size // 2:x + window_size // 2]

        # Estrai le feature utilizzando filtri di Gabor
        landmark_features = []
        for frequency in frequencies:
            for theta in orientations:
                kernel = gabor_kernel(frequency, theta=theta)
                response = np.real(scipy.signal.convolve2d(window, kernel, boundary='wrap', mode='same'))
                landmark_features.append(np.mean(response))
                landmark_features.append(np.std(response))
                landmark_features.append(np.sum(response ** 2))

        features.append(landmark_features)

    return features








