import pygame
import sounddevice as sd
import numpy as np
import time

# Paramètres audio
SAMPLE_RATE = 48000  # Fréquence d'échantillonnage
DURATION = 2  # Durée en secondes
FREQUENCY = 440  # Fréquence du son (Hz)
OUTPUT_DEVICE = 17
# 6=Digidesign Mbox2 Analog 1/2 (Di, MME (0 in, 2 out)
# 13=Digidesign Mbox2 Analog 1/2 (Digidesign Mbox 2 Audio), Windows DirectSound (0 in, 2 out)
# 15=Digidesign Mbox2 Analog 1/2 (Digidesign Mbox 2 Audio), Windows WASAPI (0 in, 2 out)
# 17=Realtek Digital Output (Realtek(R) Audio), Windows WASAPI (0 in, 2 out)
# 25=Digidesign Mbox2 Analog 1/2 (Mbox2 Out 1/2 Out), Windows WDM-KS (0 in, 2 out)

# Fonction pour générer un signal sonore
def generate_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

# Fonction pour afficher une LED rouge
def display_red_led():
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LED Rouge Simulation")
    RED = (255, 0, 0)
    screen.fill(RED)
    pygame.display.flip()
    return screen

# Fonction principale
def main():
    # Génération du son
    tone = generate_tone(FREQUENCY, DURATION, SAMPLE_RATE)

    # Configuration du pilote ASIO
    try:
        sd.default.device = OUTPUT_DEVICE  # Assure-toi que ton système détecte un pilote ASIO valide
    except Exception as e:
        print(f"Erreur lors de la configuration du pilote ASIO : {e}")
        return

    running = True
    while running:
        # Afficher la LED rouge
        screen = display_red_led()

        # Lecture du son
        sd.play(tone, samplerate=SAMPLE_RATE)
        sd.wait()  # Attend la fin de la lecture du son

        # Fermer l'affichage de la LED
        pygame.quit()

        # Pause avant le prochain affichage (facultatif)
        time.sleep(1)

    # Quitter Pygame proprement
    pygame.quit()

if __name__ == "__main__":
    main()
