"""Chaîne complète : détection, expression, affichage.

detect_faces fournit les visages. predict_expression leur donne une classe.
"""


def annotate(image, model):
    """Annote chaque visage : boîte, expression, probabilité.

    Retourne l'image annotée.
    """
    raise NotImplementedError


def annotate_video(input_path, output_path, model):
    """Annote les frames d'une courte vidéo et écrit le fichier de sortie."""
    raise NotImplementedError
