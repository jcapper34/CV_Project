"""
Description: This file contains functions to play audio of read-in sheet music
"""

import numpy as np
import simpleaudio as sa

SAMPLE_RATE = 44100
QUARTER_DURATION = 0.25
NOTE_STOP = 0.01

NOTE_FREQUENCIES = {
    "C":   [16.35, 32.70, 65.41, 130.81, 261.63, 523.25, 1046.50, 2093.00, 4186.01],
   "Db":   [17.32, 34.65, 69.30, 138.59, 277.18, 554.37, 1108.73, 2217.46, 4434.92],
    "D":   [18.35, 36.71, 73.42, 146.83, 293.66, 587.33, 1174.66, 2349.32, 4698.64],
   "Eb":   [19.45, 38.89, 77.78, 155.56, 311.13, 622.25, 1244.51, 2489.02, 4978.03],
    "E":   [20.60, 41.20, 82.41, 164.81, 329.63, 659.26, 1318.51, 2637.02],
    "F":   [21.83, 43.65, 87.31, 174.61, 349.23, 698.46, 1396.91, 2793.83],
   "Gb":   [23.12, 46.25, 92.50, 185.00, 369.99, 739.99, 1479.98, 2959.96],
    "G":   [24.50, 49.00, 98.00, 196.00, 392.00, 783.99, 1567.98, 3135.96],
   "Ab":   [25.96, 51.91, 103.83, 207.65, 415.30, 830.61, 1661.22, 3322.44],
    "A":   [27.50, 55.00, 110.00, 220.00, 440.00, 880.00, 1760.00, 3520.00],
   "Bb":   [29.14, 58.27, 116.54, 233.08, 466.16, 932.33, 1864.66, 3729.31],
    "B":   [30.87, 61.74, 123.47, 246.94, 493.88, 987.77, 1975.53, 3951.07]
}


def get_frequency(letter, octave, accidental):
    letter = letter.upper()
    if accidental == '#':
        if letter == 'G':
            letter = 'Ab'
        else:
            letter = chr(ord(letter) + 1) + 'b'
    elif accidental is not None:
        letter += accidental

    return NOTE_FREQUENCIES[letter][octave]



def play_notes(notes):
    audio = np.array([], dtype=np.int16)
    for note in notes:
        letter, octave, accidental, duration = note
        frequency = get_frequency(letter, octave, accidental)

        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), False)

        # Sine wave of note frequency
        note = np.sin(frequency * t * 2 * np.pi)

        # Ensure that highest value is in 16-bit range
        note_audio = note * (2**15 - 1) / np.max(np.abs(note))

        # Convert to 16-bit data
        note_audio = note_audio.astype(np.int16)

        audio = np.append(audio, note_audio, axis=0)    # Add note to buffer

        stop_counts = np.repeat(0, NOTE_STOP*SAMPLE_RATE)     # Add note stop time to buffer
        audio = np.append(audio, stop_counts, axis=0)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, SAMPLE_RATE)
    play_obj.wait_done()


if __name__ == '__main__':
    play_notes([
        ('E', 4, None, QUARTER_DURATION),
        ('D', 4, None, QUARTER_DURATION),
        ('C', 4, None, QUARTER_DURATION),
        ('D', 4, None, QUARTER_DURATION),
        ('E', 4, None, QUARTER_DURATION),
        ('E', 4, None, QUARTER_DURATION),
        ('E', 4, None, QUARTER_DURATION*2),
        ('D', 4, None, QUARTER_DURATION),
        ('D', 4, None, QUARTER_DURATION),
        ('D', 4, None, QUARTER_DURATION*2),
        ('E', 4, None, QUARTER_DURATION),
        ('G', 4, None, QUARTER_DURATION),
        ('G', 4, None, QUARTER_DURATION),

    ])
