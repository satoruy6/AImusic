#import subprocess
#subprocess.run(["pip3", "install", "ez_setup"])
#subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
#subprocess.run(["python", "-m", "pip", "install", "--upgrade", "setuptools"])
#subprocess.run(["apt","update"])
#subprocess.run(["apt", "install", "libfluidsynth2"])
#subprocess.run(["apt", "install", "fluid-soundfont-gm"])
#subprocess.run(["apt", "install", "build-essential"])
#subprocess.run(["apt", "install", "libasound2-dev"])
#subprocess.run(["apt", "install", "libjack-dev"])
#subprocess.run(['apt-get', 'install', 'libsndfile1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
import os
os.system("gsutil -q -m cp gs://download.magenta.tensorflow.org/models/music_vae/multitrack/* /content/")

import streamlit as st
st.set_page_config(page_title="AImusic app")
st.title("Compose Music using AI")
st.text("This site uses AI to compose a short piece of music of approximately 30 seconds.")
st.text("It takes a few minutes to compose.")
st.text("The chord progression is C-G-Am-F.")
st.text("When the download button appears, you can download the MIDI file.")
st.text("Please use it as music for posting on tiktok, youtube short video, instagram, etc.")
st.text("The following Youtube short video is an example of music composed using this AI model. Please note that the chord progression is Am-F-G-C, which is different from the chord progression you can compose on this site.")
st.text("Thank you")

st.video("https://youtu.be/aPYcGthKrXo", format="video/mp4", start_time=0)

if st.button('recompose'):
    st.experimental_rerun()

st.text("When the composition is complete, a download button will appear below.")

import numpy as np

import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta.music.sequences_lib import concatenate_sequences

import note_seq

BATCH_SIZE = 4  # 一度に扱うデータ数
Z_SIZE = 512  # 潜在変数の数
TOTAL_STEPS = 512  # コードのベクトル化に使用
CHORD_DEPTH = 49  # コードのベクトル化に使用
SEQ_TIME = 2.0  # 各NoteSequenceの長さ

def trim(seqs, seq_time=SEQ_TIME):  # NoteSequenceの長さを揃える
    for i in range(len(seqs)):
        seqs[i] = mm.extract_subsequence(seqs[i], 0.0, seq_time)
        seqs[i].total_time = seq_time

def encode_chord(chord):  # コードの文字列をベクトルに変換
    index = mm.TriadChordOneHotEncoding().encode_event(chord)
    encoded = np.zeros([TOTAL_STEPS, CHORD_DEPTH])
    encoded[0,0] = 1.0
    encoded[1:,index] = 1.0
    return encoded

def set_instruments(note_sequences):  # 楽器の調整
    for i in range(len(note_sequences)):
        for note in note_sequences[i].notes:
            if note.is_drum:
                note.instrument = 9

config = configs.CONFIG_MAP["hier-multiperf_vel_1bar_med_chords"]
model = TrainedModel(
    config,
    batch_size=BATCH_SIZE,
    checkpoint_dir_or_path="/content/model_chords_fb64.ckpt")

chord_1 = "C"
chord_2 = "G"
chord_3 = "Am"
chord_4 = "F"
chords = [chord_1, chord_2, chord_3, chord_4]

num_bars = 16
temperature = 0.2

z1 = np.random.normal(size=[Z_SIZE])
z2 = np.random.normal(size=[Z_SIZE])
z = np.array([z1+z2*t for t in np.linspace(0, 1, num_bars)])  # z1とz2の間を線形補間

seqs = [
    model.decode(
        length=TOTAL_STEPS,
        z=z[i:i+1, :],
        temperature=temperature,
        c_input=encode_chord(chords[i%4])
        )[0]
    for i in range(num_bars)
]

trim(seqs)
set_instruments(seqs)
seq = concatenate_sequences(seqs)

#mm.plot_sequence(seq)
#mm.play_sequence(seq, synth=mm.fluidsynth)

note_seq.sequence_proto_to_midi_file(seq, "AImusic.mid")  #MIDI　データに変換し保存
#st.download_button("Download midi file", open(os.path.join("AImusic.mid"), "br"), "AImusic.mid")  # ダウンロード
try:
  with open('AImusic.mid', 'rb') as f:
    content = f.read()
except FileNotFoundError:
  st.error("Error: File not Found")
except PermissionError:
  st.error("Error: Permission Denied")
st.download_button("Download midi file", content, "AImusic.mid")
