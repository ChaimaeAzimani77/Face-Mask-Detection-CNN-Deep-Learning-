# 😷 Face Mask Detection — CNN (Deep Learning)

> **Mini-Projet Deep Learning** — ENSA d'Oujda | Filière GSEIR-4 | 2025/2026  
> Classification binaire d'images de visages avec ou sans masque à l’aide d’un CNN

## 👩‍💻 Réalisé par 

- **El Azimani Chaimae**
- **Bouras Jihane**

## 📌 Problématique

Les systèmes de vision par ordinateur permettent d'automatiser la détection du port du masque dans les espaces publics.

Un **CNN standard** peut présenter des limites comme l’overfitting ou une instabilité des performances.
Ce projet propose un modèle basé sur un **Convolutional Neural Network (CNN)** capable de classifier automatiquement une image en :

- 😷 **With Mask** — Masque porté  
- ❌ **Without Mask** — Pas de masque  

avec des **résultats fiables** et une **bonne capacité de généralisation**.

## 🎯 Objectifs

- Implémenter un **CNN from scratch**
- Classifier automatiquement les images en 2 classes
- Utiliser le **Data Augmentation** pour améliorer les performances
- Évaluer le modèle via courbes, matrice de confusion et rapport de classification

## 📊 Dataset

| Split | Nombre d'images |
|---|---|
| **Entraînement (80%)** | ~5 000 images |
| **Validation (20%)** | ~1 250 images |
| **Test** | ~1 500 images |
| **Classes** | `with_mask` (1) / `without_mask` (0) |

### Exemples d'images (avec Data Augmentation)

<p align="center">
  <img src="Images/1.png" width="850"/>
</p>

## 🛠️ Partie Matérielle — Paramètres du Modèle

| Paramètre | Valeur |
|---|---|
| **Taille des images** | 128 × 128 pixels |
| **Batch size** | 32 |
| **Epochs** | 10 |
| **Optimizer** | Adam |
| **Loss function** | Binary Crossentropy |
| **Métrique** | Accuracy |
| **Modèle** | CNN |

## 💻 Partie Logicielle (Software)

### 🧾 Technologies utilisées

| Technologie | Rôle |
|---|---|
| **Python 3** | Langage principal |
| **TensorFlow / Keras** | Framework Deep Learning |
| **CNN** | Modèle de classification |
| **ImageDataGenerator** | Data Augmentation |
| **Matplotlib / Seaborn** | Visualisation |
| **Scikit-learn** | Matrice de confusion + rapport |

## ⚙️ Architecture CNN

<p align="center">
  <img src="Images/2.png" width="900"/>
</p>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Convolution + Pooling
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
