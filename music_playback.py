"""
Description: This file contains functions to play audio of read-in sheet music
"""

import numpy as np
import simpleaudio as sa

SAMPLE_RATE = 44100
QUARTER_DURATION = 0.5     # Duration of quarter note (sec)
NOTE_STOP = 0.01            # Pause time between notes (sec)

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


"""
Takes in a sequence of note groups.
-> Each note group contains a list of notes to be played at that time
-> -> Each note is represented by a tuple in the form (letter, octave, accidental, duration)
-> -> Exception: if the note is a rest, then it is just be represented by the duration value
-> -> -> Accidental should be '#' for sharp, 'b' for flat, and None if no accidental
"""
def play_notes(notes):
    audio = np.array([], dtype=np.int16)
    for note_group in notes:
        note_group = note_group if isinstance(note_group, list) else [note_group]  # Make sure it's a list

        note_wave = 0
        for part in note_group:
            part = part if isinstance(part, list) else [part]   # Make sure it's a list
            part_wave = np.array([])
            for note in part:
                if isinstance(note, tuple):
                    letter, octave, accidental, counts = note
                    duration = counts*QUARTER_DURATION
                    frequency = get_frequency(letter, octave, accidental)

                    # Create sampled time range
                    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), False)

                    # Sine wave of note frequency
                    part_wave = np.append(part_wave, np.sin(frequency * t * 2 * np.pi), axis=0)

                else: # This means it is a rest
                    duration = note
                    part_wave = np.append(part_wave, np.repeat(0, duration * SAMPLE_RATE), axis=0)

            # Add pause time between notes
            part_wave = np.append(part_wave, np.repeat(0, NOTE_STOP * SAMPLE_RATE), axis=0)

            note_wave += part_wave

        # Put everything in a 16 bit range where the max is 2^15-1
        note_audio = note_wave * (2**15 - 1) / np.max(np.abs(note_wave))

        audio = np.append(audio, note_audio.astype(np.int16), axis=0)    # Add note to buffer

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, SAMPLE_RATE)
    play_obj.wait_done()


if __name__ == '__main__':
    # Mary Had a Little Lamb
    play_notes([
        [('E', 5, None, 1), ('C', 4, None, 1)],
        [('D', 5, None, 1), ('E', 4, None, 1)],
        [('C', 5, None, 1), ('G', 4, None, 1)],
        [('D', 5, None, 1), ('E', 4, None, 1)],
        [('E', 5, None, 1), ('C', 4, None, 1)],
        [('E', 5, None, 1), ('E', 4, None, 1)],
        [('E', 5, None, 2), [('C', 4, None, 1), ('D', 4, None, 1)]],
        [('D', 5, None, 1), ('G', 3, None, 1)],
        [('D', 5, None, 1), ('B', 3, None, 1)],
        [('D', 5, None, 2), [('D', 4, None, 1), ('B', 3, None, 1)]],
        [('E', 5, None, 1), ('C', 4, None, 1)],
        [('G', 5, None, 1), ('E', 4, None, 1)],
        [('G', 5, None, 2), [('G', 4, None, 1), ('E', 4, None, 1)]],
    ])

    # # All Along the Watchtower
    # # Sharps are C, D, F, G
    # play_notes([
    #     [('F', 5, '#', QUARTER_DURATION/2), ('D', 5, '#', QUARTER_DURATION/2), ('B', 3, None, QUARTER_DURATION/2)],
    #     [('F', 5, '#', QUARTER_DURATION/2), ('D', 5, '#', QUARTER_DURATION/2), ('B', 3, None, QUARTER_DURATION/2)],
    #     [('G', 5, '#', QUARTER_DURATION/2), ('E', 5, None, QUARTER_DURATION/2), ('C', 3, '#', QUARTER_DURATION/2)],
    #     [('G', 5, '#', QUARTER_DURATION/2), ('E', 5, None, QUARTER_DURATION/2), ('C', 3, '#', QUARTER_DURATION/2)],
    #     [('G', 5, '#', QUARTER_DURATION/2), ('E', 5, None, QUARTER_DURATION/2), ('C', 3, '#', QUARTER_DURATION/2)],
    #     [('G', 5, '#', QUARTER_DURATION), ('E', 5, None, QUARTER_DURATION), ('C', 3, '#', QUARTER_DURATION)],
    # ])
