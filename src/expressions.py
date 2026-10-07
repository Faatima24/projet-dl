"""Classification des expressions faciales.

Contrat partagé avec la détection et la vidéo : tout visage, qu'il vienne
d'une image de test ou d'un crop, passe par preprocess_face avant le CNN.
"""

CLASS_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

IMAGE_SIZE = (48, 48)


def preprocess_face(image):
    """Convertit un visage en tenseur (48, 48, 1), float32, pixels / 255.

    Accepte une image niveaux de gris ou couleur (H, W) ou (H, W, C).
    """
    raise NotImplementedError


def load_datasets(data_dir):
    """Charge FER-2013 et retourne les ensembles train, validation et test.

    Le dossier test du dataset n'est pas utilisé pour construire la validation.
    """
    raise NotImplementedError


def build_mlp(input_shape=(48, 48, 1), num_classes=7):
    """MLP de référence : Flatten, Dense, Softmax."""
    raise NotImplementedError


def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    """CNN : plusieurs blocs Conv2D + ReLU + pooling, puis Dense et Softmax."""
    raise NotImplementedError


def predict_expression(crop, model):
    """Prédit l'expression d'un visage déjà extrait.

    Retourne (nom_de_classe, probabilite). Utilise preprocess_face.
    """
    raise NotImplementedError
