# 😷 Face Mask Detection — CNN (Deep Learning)

> **Mini-Projet Deep Learning** — ENSA d'Oujda | Filière GSEIR-4 | 2025/2026  
> Classification d’images de visages avec ou sans masque à l’aide d’un **CNN**

---

## 👩‍💻 Réalisé par 

- **El Azimani Chaimae**
- *(ajoute ton binôme si nécessaire)*

---

## 📌 Problématique

Dans le contexte post-pandémie, la détection automatique du port du masque est essentielle pour renforcer les mesures sanitaires.

Un modèle basé sur **CNN (Convolutional Neural Network)** est utilisé pour classifier automatiquement les images en :

- 😷 **With Mask** — Masque porté  
- ❌ **Without Mask** — Pas de masque  

Le défi principal est d’obtenir un modèle **précis**, **robuste** et capable de **généraliser**.

---

## 🎯 Objectifs

- Implémenter un **CNN from scratch**
- Classifier des images en **2 classes (Mask / No Mask)**
- Utiliser le **Data Augmentation**
- Évaluer le modèle avec :
  - Accuracy / Loss
  - Matrice de confusion
  - Rapport de classification

---

## 📊 Dataset

| Split | Description |
|---|---|
| **Entraînement** | Apprentissage du modèle |
| **Validation** | Ajustement |
| **Test** | Évaluation finale |
| **Classes** | `with_mask` / `without_mask` |

---

## 🛠️ Paramètres du Modèle

| Paramètre | Valeur |
|---|---|
| **Taille des images** | 128 × 128 |
| **Batch size** | 32 |
| **Epochs** | 10–20 |
| **Optimizer** | Adam |
| **Loss** | Binary Crossentropy |
| **Métrique** | Accuracy |

---

## 💻 Technologies utilisées

- Python 3  
- TensorFlow / Keras  
- CNN  
- ImageDataGenerator  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## ⚙️ Architecture du CNN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
