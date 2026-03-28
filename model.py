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