# Importimised
import pickle
import re
import os
import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
import os

def get_mlb_w():
  filename = 'Binarizers/mlb_w.pkl'
  if os.path.exists(filename):
    with open(filename, 'rb') as file:
      mlb_w = pickle.load(file)
  else:
    print("Ei leidnud olemasolevat sõnaliigi vektoriseerijat MultiLabelBinarizer faili nimega mlb_w.pkl")
    return None
  return mlb_w

def get_mlb_s_üldisem():
  filename = 'Binarizers/mlb_s.pkl'

  if os.path.exists(filename):
    with open(filename, 'rb') as file:
      mlb_s = pickle.load(file)
  else:
    print("Ei leidnud olemasolevat sõnaliigi vektoriseerijat MultiLabelBinarizer faili nimega mlb_s.pkl")
    return None
  return mlb_s

def tekst_vect_failid():
  # Kaust, kuhu on kogutud kõik TextVectorization'id
  kaust = "./TextVectorizations/"

  vocab_file = kaust + "text_vectorization_vocab.pkl" # TextVectorization sõnastik

  # Kui on olemas failid, siis loe sisse
  if os.path.exists(vocab_file):
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)

    tekst_vect = tf.keras.layers.TextVectorization(split="character",
                                                  output_mode="int",
                                                  output_sequence_length=20)
    tekst_vect.set_vocabulary(vocab)
  else:
    print("Ei leidnud olemasolevat TextVectorization sõnastikku text_vectorization_vocab.pkl")
    return None

  return tekst_vect

def mudel_init():
  tf.random.set_seed(7)

  # Parameetrid
  sõnavara_suurus = 67 # Sõnastiku suurus
  output_dim = 49 # Ennustatavate klasside arv
  max_pikkus = 20 # Vektori suurus

  mudel = tf.keras.models.Sequential()

  # Sisend
  mudel.add(tf.keras.layers.Input(shape=(max_pikkus, )))

  # Embedding
  mudel.add(tf.keras.layers.Embedding(input_dim=sõnavara_suurus, output_dim=output_dim, mask_zero=True))

  # LSTM
  mudel.add(tf.keras.layers.LSTM(160, return_sequences=False))

  # Dropout
  mudel.add(tf.keras.layers.Dropout(0.0))

  # Dense
  mudel.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

  # Konfigureeri
  mudel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

  # Treenitud mudeli kaalud
  mudel.load_weights("./Sonamudel/sonamudel_weights.h5")

  return mudel

def mudel_init_sõnaliigiga():
  # https://www.tensorflow.org/guide/keras/functional_api#models_with_multiple_inputs_and_outputs
  tf.random.set_seed(7)

  # Parameetrid
  sõnavara_suurus = 67 # Sõnastiku suurus
  output_dim_w = 49 # Ennustatavate klasside arv
  output_dim_s = 3 # Sõnaliikide klasside arv
  max_pikkus = 20 # Vektori suurus

  # Sisendid
  sõna_input = tf.keras.layers.Input(shape=(max_pikkus,), name='sona')
  sõnaliik_input = tf.keras.layers.Input(shape=(output_dim_s,), name='sonaliik')

  # Embeddings
  sõna_features = tf.keras.layers.Embedding(sõnavara_suurus, output_dim_w)(sõna_input)
  sõnaliik_features  = tf.keras.layers.Embedding(output_dim_s, output_dim_w)(sõnaliik_input)

  # LSTMs
  sõna_features = tf.keras.layers.LSTM(256)(sõna_features)
  sõnaliik_features = tf.keras.layers.LSTM(256)(sõnaliik_features)

  # Dropout
  sõna_features = tf.keras.layers.Dropout(0.2)(sõna_features)
  sõnaliik_features = tf.keras.layers.Dropout(0.2)(sõnaliik_features)

  # Merge
  x = tf.keras.layers.concatenate([sõna_features, sõnaliik_features])

  # Dense
  muuttüüp_pred = tf.keras.layers.Dense(output_dim_w, activation='softmax')(x)

  # Mudeli sisendid ja väljund
  mudel = tf.keras.Model(
      inputs=[sõna_input, sõnaliik_input],
      outputs=[muuttüüp_pred]
  )

  # Konfigureeri
  mudel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

  # Treenitud mudeli kaalud
  mudel.load_weights("./Sonaliigiga_sonamudel/sonaliigiga_sonamudel_weights.h5")

  return mudel

def leia_muuttyyp(sõna, sõnaliik = ''):

  df_algvormidega = pd.read_csv("andmed_algvormidega.csv", header=0, keep_default_na=False)

  # Mudeli leidmine
  try:
    if sõnaliik: # Sõnaliigiga sõnamudel
      mudel = mudel_init_sõnaliigiga()
    else: # Sõnamudel
      mudel = mudel_init()
  except Exception as e:
    print("Mudeli laadimine ebaõnnestus")
    return

  # TextVectorization
  tekst_vect = tekst_vect_failid()
  if not tekst_vect:
    print("Paiguta text_vectorization_vocab.pkl TextVectorizations kausta")
    return
  X_w = tekst_vect(sõna)

  # Muuttüübid
  mlb_w = get_mlb_w()
  if not mlb_w:
    print("Paiguta mlb_w.pkl Binarizers kausta")

  # Sõnaliigid
  if sõnaliik:
    mlb_s = get_mlb_s_üldisem()
    if not mlb_s:
      print("Paiguta mlb_s.pkl Binarizers kausta")
      return
    sõnaliik = [[s] for s in sõnaliik]
    X_s = tf.convert_to_tensor(mlb_s.transform(sõnaliik)) # Ei tohi olla Tensor (sõna) ja mitte Tensor (sõnaliik) koos, sestap convert_to_tensor

  # Ennustamine
  if sõnaliik:
    sisend_tulemused = mudel([X_w, X_s], training=False)
  else:
    sisend_tulemused = mudel(X_w, training=False)

  # Tulemuste salvestamine
  tulemused = list()
  for tulemus in sisend_tulemused:
    indeks = np.argmax(tulemus)
    ennustatud_muuttüüp = mlb_w.classes_[indeks]
    ennustatud_muuttüüp = re.sub(r"[\[\]']", "", ennustatud_muuttüüp)
    tulemused.append(ennustatud_muuttüüp)

  return tulemused

# Näited
print("Sõnamudeli tulemus sõnal \"testima\"")
print(leia_muuttyyp(["testima"]))
print("Sõnaliigiga sõnamudeli tulemus sõnal \"testima\", mille sõnaliigiks on määratud 'u' ehk muutumatu sõna")
print(leia_muuttyyp(["testima"], ["u"]))