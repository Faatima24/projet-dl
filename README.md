# Reconnaissance d’expressions faciales

Projet de **Fondamentaux du Deep Learning**

Équipe : Fatima et Nicolas  
 
## Objectif

Construire un système qui reconnaît sept expressions faciales (colère, dégoût, peur, joie, tristesse, surprise, neutre), d’abord sur un visage déjà extrait, puis sur une image ou une courte vidéo contenant plusieurs personnes.

Le problème est une **classification multiclasse** : 7 neurones de sortie, Softmax, classe prédite = probabilité la plus haute. Le jeu de test ne sert qu’à l’évaluation finale.

Dataset de travail : **FER-2013** (dépôt Kaggle `msambare/fer2013`), images 48×48 en niveaux de gris. Il n’est pas versionné dans ce dépôt.

Ordre des classes, utilisé partout :

`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

Prétraitement commun (`preprocess_face`) : niveaux de gris, 48×48, pixels en `float32` divisés par 255, entrée `(48, 48, 1)`.

## Arborescence

```text
.
├── README.md
├── requirements.txt
├── Projet_DL_Expressions_faciales (1).pdf
├── notebooks/
│   └── expressions_faciales.ipynb
├── src/
│   ├── expressions.py       # prétraitement et prédiction d’expression
│   ├── detection.py         # détection des visages
│   └── pipeline.py          # image et vidéo annotées
├── data/                    # FER-2013 en local, ignoré par git
├── models/                  # poids .keras, ignorés par git
└── reports/                 # tableau d’expériences exporté
```

Fatima mène les expressions, y compris sur chaque frame. Nicolas mène la détection. Les deux participent aux deux volets.

## Installation

Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Placer le dataset dans `data/fer2013/`, avec un dossier par classe sous `train/` et `test/` :

```text
data/fer2013/train/angry/*.jpg
data/fer2013/test/happy/*.jpg
```

Le notebook doit aussi s’exécuter sur Google Colab. Les images et les fichiers `.keras` restent hors du dépôt git.

## Livrables attendus

1. Le notebook `notebooks/expressions_faciales.ipynb`, commenté et exécutable de bout en bout.
2. Une présentation de la démarche, des architectures, des expériences et des résultats.
3. Une démonstration du modèle final (visage isolé, photo à plusieurs visages, vidéo courte si elle est prête).

## Barème

| Partie | Contenu | Points |
| --- | --- | --- |
| 1 | Données et préparation | 3 |
| 2 et 3 | MLP de référence et CNN | 5 |
| 4 et 5 | Entraînement et analyse des erreurs | 3 |
| 6 | Au moins trois expériences | 4 |
| 7 à 9 | Enrichissement, plusieurs visages, vidéo | 2 |
| Soutenance | Démonstration et maîtrise du travail | 3 |
