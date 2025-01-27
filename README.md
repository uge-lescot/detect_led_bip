# detect_led_bip
Ce script détecte et analyse les événements de synchronisation entre une LED qui s'allume dans une vidéo et un son correspondant dans la bande audio de cette vidéo. 

Voici ses principales fonctionnalités :

    Conversion vidéo vers audio : Extrait la piste audio d'une vidéo .mp4 en un fichier .wav tout en conservant les propriétés audio (comme la fréquence d'échantillonnage et le nombre de canaux).

    Détection des LED : Utilise OpenCV pour détecter les moments où une LED rouge s'allume dans les frames vidéo. Ce traitement peut être accéléré avec le GPU via cuPy (si disponible).

    Détection des sons : Identifie les fronts montants dans la forme d'onde audio (origine des bips) à l'aide de la dérivée discrète et d'un seuil adaptatif.

    Calcul des délais : Calcule les écarts temporels entre les timestamps des LED et les sons détectés, en avertissant des anomalies comme des délais négatifs ou trop longs.

    Enregistrement des résultats :
        Sauvegarde les données dans un fichier Excel avec des miniatures des frames correspondantes aux LED.
        Enregistre un graphe montrant la forme d'onde audio et les détections des LED et des bips.
        Exporte un résumé des résultats (moyenne et écart-type des délais) dans un fichier texte.

    Traitement par lots : Prend en charge plusieurs vidéos placées dans un répertoire et organise les résultats dans des sous-dossiers horodatés.

Points techniques

    Multi-threading : Utilise plusieurs threads pour accélérer l'analyse vidéo.
    CUDA (GPU) : Optimise les traitements de détection des LED si un GPU compatible est disponible.
    Seuil adaptatif pour les sons : Le seuil pour détecter les fronts montants audio est basé sur l'écart-type de la dérivée audio pour s'adapter aux variations du signal.

Exemple d'utilisation

Le script traite automatiquement toutes les vidéos .mp4 dans le répertoire data/videos, génère les résultats dans data/results/<horodatage> et affiche des tracés pour validation visuelle.
