from datetime import datetime
import cv2
import numpy as np
import os
import wave
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenPyxlImage
from PIL import Image
from threading import Lock
import subprocess

# Fonction pour vérifier la disponibilité de CUDA
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def convert_video_to_audio(input_video_path, output_audio_path):
    """
    Convertit un fichier vidéo .mp4 en fichier audio .wav tout en conservant les propriétés audio d'origine.

    Parameters:
        input_video_path (str): Chemin du fichier vidéo .mp4 en entrée.
        output_audio_path (str): Chemin du fichier audio .wav en sortie.

    Returns:
        None
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Le fichier vidéo '{input_video_path}' est introuvable.")

    if not os.path.exists(output_audio_path):
        # Construire la commande FFmpeg
        command = [
            "ffmpeg", "-i", input_video_path,  # Entrée vidéo
            "-vn",  # Désactiver la piste vidéo
            "-acodec", "pcm_s16le",  # Codec audio WAV (PCM 16 bits)
            "-r", "original",  # Conserver la fréquence d'échantillonnage d'origine
            "-ac", "2",  # Conserver le nombre de canaux d'origine (stéréo)
            output_audio_path  # Fichier de sortie
        ]

        try:
            # Exécuter la commande FFmpeg
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Conversion réussie : '{input_video_path}' -> '{output_audio_path}'")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la conversion : {e.stderr.decode()}")
            raise


# Détection des LED dans une plage de frames avec GPU
def process_frame_range(video_path, start_frame, end_frame, pbar, lock):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    led_timestamps = []

    # Définir la plage pour détecter le rouge en HSV
    lower_red1 = cp.array([0, 100, 100], dtype=np.uint8) if CUDA_AVAILABLE else np.array([0, 100, 100], dtype=np.uint8)
    upper_red1 = cp.array([10, 255, 255], dtype=np.uint8) if CUDA_AVAILABLE else np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = cp.array([160, 100, 100], dtype=np.uint8) if CUDA_AVAILABLE else np.array([160, 100, 100], dtype=np.uint8)
    upper_red2 = cp.array([179, 255, 255], dtype=np.uint8) if CUDA_AVAILABLE else np.array([179, 255, 255], dtype=np.uint8)

    previous_led_state = False  # État initial : LED éteinte

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir l'image en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if CUDA_AVAILABLE:
            # Traitement sur GPU
            hsv_gpu = cp.asarray(hsv)
            mask1 = cp.logical_and(hsv_gpu >= lower_red1, hsv_gpu <= upper_red1).all(axis=2)
            mask2 = cp.logical_and(hsv_gpu >= lower_red2, hsv_gpu <= upper_red2).all(axis=2)
            mask = cp.logical_or(mask1, mask2)
            current_led_state = cp.count_nonzero(mask) > 20
        else:
            # Traitement sur CPU
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            current_led_state = cv2.countNonZero(mask) > 20

        # Ajouter un timestamp uniquement lors d'une transition "éteinte" → "allumée"
        if current_led_state and not previous_led_state:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            led_timestamps.append((timestamp, frame))

        previous_led_state = current_led_state

        # Mettre à jour la barre de progression en utilisant un verrou
        with lock:
            pbar.update(1)

    cap.release()
    return led_timestamps


# Détection des LED avec GPU et multi-thread
def detect_led(video_path, output_file, num_threads=4):
    # # Si le fichier Excel existe déjà, ne pas recalculer
    # if os.path.exists(output_file):
    #     print(f"Les timestamps des LEDs sont déjà enregistrés dans {output_file}. Étape ignorée.")
    #     return None

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_per_thread = frame_count // num_threads
    ranges = [
        (i * frames_per_thread, (i + 1) * frames_per_thread)
        for i in range(num_threads)
    ]
    ranges[-1] = (ranges[-1][0], frame_count)

    led_data = []

    # Barre de progression partagée et verrou
    lock = Lock()
    pbar = tqdm(total=frame_count, desc=f"Analyse vidéo {os.path.basename(video_path)}", unit="frame")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_frame_range, video_path, start, end, pbar, lock)
            for start, end in ranges
        ]

        for future in futures:
            led_data.extend(future.result())

    pbar.close()
    led_data.sort(key=lambda x: x[0])
    return led_data


# Détection des sons
def detect_sounds(audio_path, led_timestamps, output_dir):
    # Charger le fichier audio en utilisant wave
    with wave.open(audio_path, 'r') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate
        audio_data = wav_file.readframes(n_frames)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    # Normaliser les échantillons audio
    audio_samples /= np.max(np.abs(audio_samples))
    time_axis = np.linspace(0, duration, len(audio_samples))

    # Calcul de la dérivée discrète pour détecter les transitions
    derivative = np.diff(audio_samples)
    adaptive_threshold = 0.8 * np.std(derivative)  # Seuil = x fois l'écart-type
    threshold = adaptive_threshold  # Seuil adaptatif pour considérer un front montant
    # threshold = 0.15  # Seuil fixe pour considérer un front montant
    silence_duration = 1  # Durée minimale entre deux fronts en secondes
    min_distance = int(sample_rate * 2 * silence_duration)

    # Détection des fronts montants
    peaks = np.where(derivative > threshold)[0]  # Indices des variations significatives
    sound_timestamps = []

    if len(peaks) > 0:
        # Filtrer pour garder uniquement les premiers points significatifs dans une montée
        last_peak = -min_distance
        for peak in peaks:
            if peak - last_peak >= min_distance:
                timestamp = peak / (sample_rate * 2)
                sound_timestamps.append(timestamp)
                last_peak = peak

    # Tracer le signal audio avec les fronts montants détectés
    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, audio_samples, label="Signal Audio", alpha=0.8)
    plt.scatter(sound_timestamps, [audio_samples[int(t * sample_rate)] for t in sound_timestamps],
                color="red", marker="X", label="Origines des fronts montants")
    for led_time in led_timestamps:
        plt.axvline(x=led_time, color="green", linestyle="--", label="LED Timestamp" if led_time == led_timestamps[0] else "")

    plt.title("Signal Audio avec Origines des Fronts Montants et LED Timestamps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Enregistrer le tracé
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"audio_led_fronts.png")
    plt.savefig(plot_path)

    # Afficher le tracé et mettre en pause jusqu'à appui sur la barre d'espace
    plt.show()
    input("Appuyez sur 'Entrée' pour continuer...")
    plt.close()

    print("Analyse audio : Origines des fronts montants détectées.")
    return sound_timestamps


# Calcul des délais entre LED et sons
def calculate_delays(led_data, sound_timestamps, max_diff=0.1):
    delays = []
    for led_time, _ in led_data:
        closest_sound = min(sound_timestamps, key=lambda t: abs(t - led_time))
        delay = closest_sound - led_time
        delays.append(delay)

        if abs(delay) > max_diff:
            print(
                f"⚠️ Avertissement : Écart de {delay:.3f}s entre LED ({led_time:.3f}s) et son ({closest_sound:.3f}s) dépasse {max_diff:.3f}s"
            )
        if delay < 0:
            print(
                f"❌ Erreur : Délai négatif détecté ({delay:.3f}s). La LED ({led_time:.3f}s) est censée s'allumer avant le son ({closest_sound:.3f}s)."
            )

    mean_delay = statistics.mean(delays)
    std_delay = statistics.stdev(delays)
    return delays, mean_delay, std_delay


# Enregistrement des résultats avec Excel
def save_to_excel_with_thumbnails(led_data, sound_timestamps, delays, output_file, thumbnail_dir):
    wb = Workbook()
    ws = wb.active
    ws.title = "Résultats"
    ws.append(["LED Timestamps", "Sound Timestamps", "Delays (s)", "Thumbnail"])

    os.makedirs(thumbnail_dir, exist_ok=True)

    for i, ((led_time, frame), delay) in enumerate(zip(led_data, delays)):
        thumbnail_path = os.path.join(thumbnail_dir, f"thumbnail_{i}.png")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((150, 150))
        image.save(thumbnail_path)

        sound_time = sound_timestamps[i] if i < len(sound_timestamps) else None
        ws.append([led_time, sound_time, delay])

        img = OpenPyxlImage(thumbnail_path)
        ws.add_image(img, f"D{i+2}")

    wb.save(output_file)


# Enregistrement des résultats dans un fichier texte
def save_results_to_text(mean_delay, std_delay, output_file):
    with open(output_file, 'w') as f:
        f.write("Résultats de l'analyse des délais LED/Sons\n")
        f.write(f"Moyenne des délais : {mean_delay:.3f} secondes\n")
        f.write(f"Écart-type des délais : {std_delay:.3f} secondes\n")
    print(f"Résultats enregistrés dans {output_file}")


def debug_timestamps(led_timestamps, sound_timestamps):
    for i, (led_time, sound_time) in enumerate(zip(led_timestamps, sound_timestamps)):
        if sound_time < led_time:
            print(f"⚠️ Incohérence : Son détecté avant la LED ({sound_time:.3f}s < {led_time:.3f}s)")


# Traitement de plusieurs vidéos
def process_videos(video_paths, data_dir, date_time, max_diff=0.1):
    for video_name in video_paths:
        video_path = os.path.join(data_dir, "videos", video_name)
        audio_path = video_path.replace(".mp4", ".wav")
        convert_video_to_audio(str(video_path), str(audio_path))

        video_identifier = os.path.splitext(video_name)[0]
        results_dir = str(os.path.join(data_dir, "results", date_time, video_identifier))
        output_file = os.path.join(results_dir, f"{video_identifier}_results.xlsx")

        led_data = detect_led(video_path, output_file, num_threads=4)
        led_timestamps, _ = zip(*led_data)

        sound_timestamps = detect_sounds(audio_path, led_timestamps, results_dir)
        delays, mean_delay, std_delay = calculate_delays(led_data, sound_timestamps, max_diff)
        save_to_excel_with_thumbnails(led_data, sound_timestamps, delays, output_file, results_dir)
        print(f"Résultats pour {video_name}: Moyenne = {mean_delay:.3f}s, Écart-type = {std_delay:.3f}s")

        # Enregistrement des résultats
        text_output_file = os.path.join(results_dir, f"{video_identifier}_results.txt")
        save_results_to_text(mean_delay, std_delay, text_output_file)

# Liste de vidéos à traiter
data_directory = "data"
video_list = [f for f in os.listdir(os.path.join(data_directory, "videos")) if f.endswith('.mp4')]
date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
process_videos(video_list, data_directory, date_time)
