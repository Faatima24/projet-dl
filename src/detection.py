"""Détection des visages dans une image.

Un modèle YOLO entraîné sur COCO localise des personnes, pas des visages.
Le détecteur retenu doit être un modèle de visages, ou une autre méthode
justifiée dans le notebook.
"""


def detect_faces(image):
    """Localise les visages.

    Retourne une liste de dictionnaires :
    {"box": (x1, y1, x2, y2), "crop": image_du_visage, "detection_score": float}

    Le crop est brut. La normalisation est faite ensuite par preprocess_face.
    Le score de détection n'est pas la probabilité d'expression.
    """
    raise NotImplementedError
