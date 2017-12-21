import random, re
from collections import defaultdict
# freqDict is a dict of dict containing frequencies
def addToProbDict(fileName, freqs):
	f = open(fileName, 'r')
	words = re.sub("\n", " \n", f.read()).lower().split(' ')
	# count the frequencies of each word to their successors
	for curr_word, succ_word in zip(words[1:], words[:-1]):
		freqs.setdefault(curr_word, {succ_word: 1})
		freqs[curr_word].setdefault(succ_word, 0)
		freqs[curr_word][succ_word] += 1
	# compute percentages
	prob_dict = defaultdict()
	for curr_word, succ_freqs in freqs.items():
		prob_dict[curr_word] = {}
		curr_total = sum(succ_freqs.values())
		for succ_word in succ_freqs:
			prob_dict[curr_word][succ_word] = succ_freqs[succ_word] / curr_total
	return prob_dict


def nextWord(currWord, probDict):
	if currWord not in probDict:
		return random.choice(list(probDict.keys()))

	succ_probs = probDict[currWord]
	rand_prob = random.random()
	curr_prob = 0.0
	for succ_word in succ_probs:
		curr_prob += succ_probs[succ_word]
		if rand_prob <= curr_prob:
			return succ_word
	return random.choice(list(probDict.keys()))


def rap(startWord, probDict, T = 50):
	new_rap = [startWord]
	for t in range(T):
		new_rap.append(nextWord(new_rap[-1], probDict))
	return " ".join(new_rap)


word_to_successor_freqs = defaultdict()
word_to_successor_probs = addToProbDict('lyrics1.txt', word_to_successor_freqs)
word_to_successor_probs = addToProbDict('lyrics2.txt', word_to_successor_freqs)

first_word = input("What do you want to start your rap with?\n > ")
print("Alright, here's your rap:")
print(rap(first_word, word_to_successor_probs))
