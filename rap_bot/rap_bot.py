# basic dependencies
import io
import sys
from collections import defaultdict
# keras dependencies
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
# numpy depencency
import numpy as np

SEQ_LEN = 40
SEQ_STEP = 3

EPOCH_TIME = 20
DIVERSITY = 1.0

SRC = "kanye.txt"

# Retrieve unique characters from source
text = io.open(SRC, 'r', encoding = 'utf8').read().lower()
TEXT_LEN = len(text)
chars = sorted(list(set(text)))
CHARS_LEN = len(chars)

# sequences of lyrics chunk; next chars as training labels
seqs, next_chars = [], []
for i in range(0, TEXT_LEN - SEQ_LEN, SEQ_STEP):
  seqs.append(text[i : i + SEQ_LEN])
  next_chars.append(text[i + SEQ_LEN])

char_to_index = defaultdict()
index_to_char = defaultdict()
for i, c in enumerate(chars):
  char_to_index[c] = i
  index_to_char[i] = c

# X,y are sequences and next_chars as vectors
X = np.zeros((len(seqs), SEQ_LEN, CHARS_LEN), dtype = np.bool)
y = np.zeros((len(seqs), CHARS_LEN), dtype = np.bool)
for i, s in enumerate(seqs):
  for j, c in enumerate(s):
    X[i, j, char_to_index[c]] = 1
  y[i, char_to_index[next_chars[i]]] = 1

# Build model
model = Sequential()
model.add(LSTM(128, input_shape = (SEQ_LEN, CHARS_LEN)))
model.add(Dense(CHARS_LEN))
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

# Train model
model.fit(X, y, batch_size = 128, epochs = EPOCH_TIME)
# model = load_model("") # When I have trained weights

sentence = "The grass is greener on the other side o"
sentence = sentence.lower()
generated_lyrics = sentence

for i in range(400):
  x = np.zeros((1, SEQ_LEN, len(chars)))
  for t, char in enumerate(sentence):
    x[0, t, char_to_index[char]] = 1.0
  
  def getPredictions(temp = 1):
    temp = temp if temp != 0 else 1
    pred_list = model.predict(x, verbose = 0)[0]
    pred_arr = np.asarray(pred_list).astype('float64')
    logged_pred = [np.log(p) / temp for p in pred_arr]
    exp_sum = sum([np.exp(lp) for lp in logged_pred])
    preds = [np.exp(lp) / exp_sum for lp in logged_pred]
    return preds
  
  # get probabilities of predictions in multinomial distribution
  predictions = getPredictions(DIVERSITY)
  mult_probs = np.random.multinomial(1, predictions, 1)
  
  next_index = np.argmax(mult_probs)
  next_char = index_to_char[next_index]

  generated_lyrics += next_char
  sentence = sentence[1:] + next_char

  sys.stdout.write(next_char)
  sys.stdout.flush()

with open("generated_rap.txt", "w") as text_file:
  print(generated_lyrics, file = text_file)
  