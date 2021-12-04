import muda
import jams
import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

def augment_data(audio_path):
    audio_path = './data/genres_original/blues/blues.00000.wav'
    j_orig = muda.load_jam_audio(jams.JAMS(), audio_path)
    pitch_shift = muda.deformers.RandomPitchShift(n_samples=3)
    time_stretch = muda.deformers.RandomTimeStretch(n_samples=3)
    pipeline = muda.Union(steps=[('pitch_shift', pitch_shift), ('time_stretch', time_stretch)])

    output_path = os.path.join('./data/genres_data_augm', audio_path.replace('./data/genres_original/', ''))
    for i, j_new in enumerate(pipeline.transform(j_orig)):
        muda.save(output_path.replace('.wav', '{}.wav'.format(i)), '', j_new)


def compute_melgram(audio_path):
    # mel-spectrogram parameters
    N_FFT = 2048
    N_MELS = 128
    HOP_LEN = 1024

    y, sr = librosa.load(audio_path, mono=True)
    sg, _ = librosa.effects.trim(y) # trim silent edges

    S = librosa.feature.melspectrogram(sg, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=HOP_LEN)
    plt.show()
