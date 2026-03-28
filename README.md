# ⚡ ÉnergiAI — Estimation de Consommation Énergétique

> Modèle d'intelligence artificielle simple pour estimer la consommation énergétique d'un bâtiment à partir de la température extérieure et du nombre d'occupants.

![HTML](https://img.shields.io/badge/HTML-E34F26?style=flat&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Vercel](https://img.shields.io/badge/Déployé%20sur-Vercel-000000?style=flat&logo=vercel&logoColor=white)

---

## 🎯 Objectif

L'objectif de ce projet est de créer un modèle simple d'intelligence artificielle capable d'**estimer la consommation énergétique d'un bâtiment** à partir de deux paramètres clés :

| Variable | Description |
|----------|-------------|
| 🌡️ `x₁` — Température extérieure | Influence le chauffage et la climatisation |
| 👥 `x₂` — Occupation du bâtiment | Influence l'utilisation des équipements |

---

## 🤖 Le Modèle IA — Régression Linéaire

Le modèle utilise une **régression linéaire** entraînée par **descente de gradient** :

```
y = w₁·x₁ + w₂·x₂ + b
```

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `w₁` | 2.8 | Chaque °C ajoute 2.8 kWh |
| `w₂` | 1.5 | Chaque personne ajoute 1.5 kWh |
| `b`  | 80   | Consommation de base du bâtiment |

### Exemple de prédiction

```
Température = 15°C, Occupation = 50 personnes
y = 2.8 × 15 + 1.5 × 50 + 80 = 42 + 75 + 80 = 197 kWh
```

---

## 🚀 Démo en ligne

🔗 **[Voir la démo sur Vercel →](https://your-project.vercel.app)**

---

## 📁 Structure du projet

```
energiai/
│
├── index.html        # Application interactive complète (HTML/CSS/JS)
├── model.py          # Modèle IA en Python (régression linéaire)
└── README.md         # Documentation du projet
```

---

## 🖥️ Interface Interactive

L'interface web permet de :

- Ajuster la **température** (-10°C à +40°C) via un slider
- Ajuster l'**occupation** (0 à 200 personnes) via un slider
- Voir la **prédiction en temps réel** en kWh
- Visualiser la **décomposition** de la consommation (base / chauffage / occupation)
- Lire la **formule mathématique** mise à jour dynamiquement
- Consulter le **code Python** du modèle directement dans l'interface

---

## 🐍 Code Python du modèle

```python
import numpy as np

# Génération de données synthétiques
np.random.seed(42)
n_samples = 200

temperature = np.random.uniform(-10, 40, n_samples)
occupation  = np.random.uniform(0, 200, n_samples)

# Consommation réelle avec bruit
w1_reel, w2_reel, b_reel = 2.8, 1.5, 80
bruit = np.random.normal(0, 10, n_samples)
consommation = w1_reel * temperature + w2_reel * occupation + b_reel + bruit


class ModeleEnergie:
    def __init__(self):
        self.w1 = 0.0
        self.w2 = 0.0
        self.b  = 0.0

    def predire(self, temp, occ):
        return self.w1 * temp + self.w2 * occ + self.b

    def entrainer(self, X1, X2, Y, lr=0.0001, epochs=1000):
        n = len(Y)
        for epoch in range(epochs):
            Y_pred = self.w1 * X1 + self.w2 * X2 + self.b
            erreur = Y_pred - Y

            dw1 = (2/n) * np.dot(erreur, X1)
            dw2 = (2/n) * np.dot(erreur, X2)
            db  = (2/n) * np.sum(erreur)

            self.w1 -= lr * dw1
            self.w2 -= lr * dw2
            self.b  -= lr * db


# Entraînement
modele = ModeleEnergie()
modele.entrainer(temperature, occupation, consommation)

# Prédiction
prediction = modele.predire(15, 50)
print(f"Consommation estimée : {prediction:.1f} kWh")
```

---

## ⚙️ Installation & Lancement en local

### Option 1 — Ouvrir directement dans le navigateur

```bash
# Cloner le repo
git clone https://github.com/ton-username/energiai.git
cd energiai

# Ouvrir index.html dans ton navigateur
open index.html        # macOS
start index.html       # Windows
xdg-open index.html    # Linux
```

### Option 2 — Serveur local Python

```bash
cd energiai
python -m http.server 8000
# Ouvrir : http://localhost:8000
```

### Option 3 — Exécuter le modèle Python

```bash
pip install numpy
python model.py
```

---

## ☁️ Déploiement sur Vercel

```bash
# Installer Vercel CLI
npm install -g vercel

# Déployer
cd energiai
vercel
```

Ou déposer directement `index.html` sur [vercel.com/new](https://vercel.com/new).

---

## 📊 Niveaux de consommation

| Consommation | Niveau | Interprétation |
|---|---|---|
| < 150 kWh | 🟢 Faible | Bâtiment peu sollicité |
| 150 – 300 kWh | 🟡 Modérée | Utilisation normale |
| 300 – 450 kWh | 🟠 Élevée | Forte occupation ou température extrême |
| > 450 kWh | 🔴 Très élevée | Conditions critiques |

---

## 📚 Concepts utilisés

- **Régression linéaire** — modèle supervisé de prédiction
- **Descente de gradient** — algorithme d'optimisation des poids
- **MSE (Mean Squared Error)** — fonction de coût minimisée
- **Données synthétiques** — générées avec NumPy pour simuler un vrai jeu de données

---

## 👤 Auteur

Projet réalisé dans le cadre d'un cours d'intelligence artificielle.

---

## 📄 Licence

MIT License — libre d'utilisation et de modification.
