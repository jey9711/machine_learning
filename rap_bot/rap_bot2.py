import pronouncing
import markovify
import re
import random
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers.core import Dense

depth = 4
max_syllables = 16
train_mode = False
artist = "kanye_west"
rap_file = "generated_rap.txt"

def createNetwork(depth):
	model = Sequential()
	model.add(LSTM(4, input_shape = (2, 2), return_sequences = True))
	for i in range(depth):
		model.add(LSTM(8, return_sequences = True))
	model.add(LSTM(2, return_sequences = True))
	model.summary()
	model.compile(optimizer = 'rmsprop', loss = 'mse')
	if not train_mode and artist + ".rap" in os.listdir("."):
		model.load_weights(str(artist + ".rap"))
    print("loading saved network: " + str(artist) + ".rap")
	return model

def createMarkovModel(fileName):
  f = open(fileName, "r")
	read = f.read()
	text_model = markovify.NewlineText(read)
	return text_model


def countSyllables(line):
	total_syllables = 0
  vowels_plus_y = "aeiouy"
  words = line.split(" ")
	for word in words:
    n = len(word)
    word_syllables = 0
    for i in range(n) :
      if i == 0 and word[i] in vowels_plus_y :
        word_syllables = word_syllables + 1
      elif word[i - 1] not in vowels_plus_y :
        if (i < len(word) - 1 or i == len(word) - 1) and 
          word[i] in vowels_plus_y):
          word_syllables = word_syllables + 1
    if len(word) > 0 and word_syllables == 0 :
      word_syllables = 1
    total_syllables += word_syllables
	return total_syllables

def rhymeIndex(lyrics, isTraining):
	if not isTraining and str(artist) + ".rhymes" in os.listdir("."):
		print("loading saved rhymes from " + str(artist) + ".rhymes")
		return open(str(artist) + ".rhymes", "r").read().split("\n")
  rhyme_master_list = []
  print("Building list of all the rhymes")
  for i in lyrics:
    word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()
    rhymeslist = pronouncing.rhymes(word)
    rhymeslist = [x.encode('UTF8') for x in rhymeslist]
    rhymeslistends = []
    for i in rhymeslist:
      rhymeslistends.append(i[-2:])
    
    try: rhymescheme = max(set(rhymeslistends), key = rhymeslistends.count)
    except Exception: rhymescheme = word[-2:]
    rhyme_master_list.append(rhymescheme)
  rhyme_master_list = list(set(rhyme_master_list))

  reverselist = [x[::-1] for x in rhyme_master_list]
  reverselist = sorted(reverselist)
  
  rhymelist = [x[::-1] for x in reverselist]

  f = open(str(artist) + ".rhymes", "w")
  f.write("\n".join(rhymelist))
  f.close()
  print rhymelist
  return rhymelist

def rhyme(line, rhyme_list):
	word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
	rhymeslist = pronouncing.rhymes(word)
	rhymeslist = [x.encode('UTF8') for x in rhymeslist]
	rhymeslistends = []
	for i in rhymeslist:
		rhymeslistends.append(i[-2:])
	try:
		rhymescheme = max(set(rhymeslistends), key = rhymeslistends.count)
	except Exception:
		rhymescheme = word[-2:]
	try:
		float_rhyme = rhyme_list.index(rhymescheme)
		float_rhyme = float_rhyme / float(len(rhyme_list))
		return float_rhyme
	except Exception:
		return None


def split_lyrics_file(text_file):
	text = open(text_file).read()
	text = text.split("\n")
	while "" in text:
		text.remove("")
	return text


def generateRap(text_model, text_file):
	bars = []
	last_words = []
	lyriclength = len(open(text_file).read().split("\n"))
	count = 0
	markov_model = createMarkovModel(text_file)
	
	while len(bars) < lyriclength / 9 and count < lyriclength * 2:
		bar = markov_model.make_sentence()

		if type(bar) != type(None) and syllables(bar) < max_syllables:
			
			def get_last_word(bar):
				last_word = bar.split(" ")[-1]
				if last_word[-1] in "!.?,":
					last_word = last_word[:-1]
				return last_word
				
			last_word = get_last_word(bar)
			if bar not in bars and last_words.count(last_word) < 3:
				bars.append(bar)
				last_words.append(last_word)
				count += 1
	return bars

def build_dataset(lines, rhyme_list):
	dataset = []
	line_list = []
	for line in lines:
		line_list = [line, syllables(line), rhyme(line, rhyme_list)]
		dataset.append(line_list)
	
	x_data = []
	y_data = []
	
	for i in range(len(dataset) - 3):
		line1 = dataset[i][1:]
		line2 = dataset[i + 1][1:]
		line3 = dataset[i + 2][1:]
		line4 = dataset[i + 3][1:]

		x = [line1[0], line1[1], line2[0], line2[1]]
		x = np.array(x)
		x = x.reshape(2,2)
		x_data.append(x)

		y = [line3[0], line3[1], line4[0], line4[1]]
		y = np.array(y)
		y = y.reshape(2,2)
		y_data.append(y)
		
	x_data = np.array(x_data)
	y_data = np.array(y_data)
	
	return x_data, y_data
	
def generateRap(lines, rhyme_list, lyrics_file, model):
	rap_vectors = []
	human_lyrics = split_lyrics_file(lyrics_file)
	
	initial_index = random.choice(range(len(human_lyrics) - 1))
	initial_lines = human_lyrics[initial_index:initial_index + 2]
	
	starting_input = []
	for line in initial_lines:
		starting_input.append([syllables(line), rhyme(line, rhyme_list)])

	starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(1, 2, 2))
	rap_vectors.append(starting_vectors)
	
	for i in range(100):
    last_vector = np.array([rap_vectors[-1]]).flatten().reshape(1, 2, 2)
		rap_vectors.append(model.predict(last_vector))
	
	return rap_vectors
	
def vectors_into_song(vectors, generated_lyrics, rhyme_list):
	print("\n\n")
	print("About to write rap (this could take a moment)...")
	print("\n\n")
	def last_word_compare(rap, line2):
		penalty = 0 
		for line1 in rap:
			word1 = line1.split(" ")[-1]
			word2 = line2.split(" ")[-1]
			 
			while word1[-1] in "?!,. ":
				word1 = word1[:-1]
			
			while word2[-1] in "?!,. ":
				word2 = word2[:-1]
			
			if word1 == word2:
				penalty += 0.2
				
		return penalty

	def calculate_score(vector_half, syllables, rhyme, penalty):
		desired_syllables = vector_half[0]
		desired_rhyme = vector_half[1]
		desired_syllables = desired_syllables * max_syllables
		desired_rhyme = desired_rhyme * len(rhyme_list)
		score = 1.0 - (abs((float(desired_syllables) - float(syllables))) + abs((float(desired_rhyme) - float(rhyme)))) - penalty
		return score
		
	dataset = []
	for line in generated_lyrics:
		line_list = [line, syllables(line), rhyme(line, rhyme_list)]
		dataset.append(line_list)
	
	rap = []
	vector_halves = []
	
	for vector in vectors:
		vector_halves.append(list(vector[0][0])) 
		vector_halves.append(list(vector[0][1]))
		
	for vector in vector_halves:
		scorelist = []
		for item in dataset:
			line = item[0]
			
			if len(rap) != 0:
				penalty = last_word_compare(rap, line)
			else:
				penalty = 0
			total_score = calculate_score(vector, item[1], item[2], penalty)
			score_entry = [line, total_score]
			scorelist.append(score_entry)
		
		fixed_score_list = []
		for score in scorelist:
			fixed_score_list.append(float(score[1]))
		max_score = max(fixed_score_list)
		for item in scorelist:
			if item[1] == max_score:
				rap.append(item[0])
				print(item[0])
				
				for i in dataset:
					if item[0] == i[0]:
						dataset.remove(i)
						break
				break
	return rap

def train(xData, yData, model):
	model.fit(np.array(xData), np.array(yData), batch_size = 2, epochs = 5, verbose = 1)
	model.save_weights(artist + ".rap")
			  
def main(depth, isTraining):
	model = create_network(depth)
	text_file = "lyrics.txt"
	text_model = createMarkovModel(text_file)
	
	if isTraining:
		bars = split_lyrics_file(text_file)
    rhyme_list = rhymeIndex(bars)
    x_data, y_data = build_dataset(bars, rhyme_list)
		train(x_data, y_data, model)
	else:
		bars = generateRap(text_model, text_file)
    rhyme_list = rhymeIndex(bars)
    vectors = generateRap(bars, rhyme_list, text_file, model)
		rap = vectors_into_song(vectors, bars, rhyme_list)
		f = open(rap_file, "w")
		for bar in rap:
			f.write(bar)
			f.write("\n")
		
main(depth, train_mode)
