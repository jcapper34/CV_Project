"""
Description: This file contains functions to play audio of read-in sheet music
"""

import numpy as np
import simpleaudio as sa
import wave


SAMPLE_RATE = 44100
COUNT_DURATION = 0.4                # Duration of quarter note (sec)
BETWEEN_NOTE_REST = 0.01            # Pause time between notes (sec)

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


# Find frequency of note
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


# Plays back a list of staff notes
# Assumes that staffs are grouped in pairs
def play_sheet(staffs, group=2):
    audio_buffer = np.array([])
    for i in range(0, len(staffs), group):
        row_buffer = 0
        for staff in staffs[i:i+group]:
            staff_buffer = create_notes_buffer(staff.notes)     # Get notes buffer
            # Add rest so that buffers are same length
            if isinstance(row_buffer, np.ndarray):
                size_d = len(row_buffer) - len(staff_buffer)
                if size_d > 0:
                    staff_buffer = np.append(staff_buffer, np.repeat(0, size_d, axis=0))
                elif size_d < 0:
                    row_buffer = np.append(row_buffer, np.repeat(0, -1*size_d, axis=0))

            row_buffer += staff_buffer

        row_buffer = row_buffer * (2 ** 15 - 1) / np.max(np.abs(row_buffer))    # Scale down to 16 bits

        audio_buffer = np.append(audio_buffer, row_buffer.astype(np.int16), axis=0)     # Add row to audio buffer

    play_obj = sa.play_buffer(audio_buffer.astype(np.int16), 1, 2, SAMPLE_RATE)
    play_obj.wait_done()


# Creates an audio buffer for a list of notes
def create_notes_buffer(notes, play=False):
    audio_buffer = np.array([])
    for value, _ in notes:
        if isinstance(value, tuple):    # This means it is a note
            letter, annotation, octave, counts = value
            duration = counts * COUNT_DURATION
            frequency = get_frequency(letter, octave, annotation)
        else:   # This means it is a rest
            counts = value
            duration = counts * COUNT_DURATION
            frequency = 0

        # Time array
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), False)

        # Sine wave of note frequency
        audio_buffer = np.append(audio_buffer, np.sin(frequency * t * 2 * np.pi), axis=0)

        # Add small rest between notes
        audio_buffer = np.append(audio_buffer, np.repeat(0, BETWEEN_NOTE_REST*SAMPLE_RATE), axis=0)

    if play:
        audio_buffer *= (2 ** 15 - 1) / np.max(np.abs(audio_buffer))    # Scale down to 16 bits
        play_obj = sa.play_buffer(audio_buffer.astype(np.int16), 1, 2, SAMPLE_RATE)
        play_obj.wait_done()

    return audio_buffer
