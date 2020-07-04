### RNN will take in text from author --> generate text in same style
# Recurrent neural networks good for sequential data (in time)
# ex) music, text, stock prices, etc.
# Datapoints have impact on subsequent datapoints (letters, notes, etc)
# there are RNNs, LSTM, GRU

### RNN will take in text from author --> generate text in same style
# Recurrent neural networks good for sequential data (in time)
# ex) music, text, stock prices, etc.
# Datapoints have impact on subsequent datapoints (letters, notes, etc)
# there are RNNs, LSTM, GRU

import numpy as np
import random
import keras



#=====Formatting Dataset-----#

### import Sun Tzu's Art of War - want a lot of text for training
# 'with' makes it cleaner to deal with opening files
filename = 'artofwar.txt'
with open(filename, encoding="utf8") as f: # import text file
    text = [x.lower() for x in f.read()] # set all text to lowercase so RNN doesn't have to learn capitalization rules
print('loaded file')
### Convert characters into numbers for neural networks (one-hot encoding)
# set(text) makes a set of all unique characters in string
# ex) set(pythonpythonoonnppyy) --> {'p', 'y', 't', 'h', 'o', 'n'} note: letters in set are not in any particular order from set()
# list(set(text)) converts set into list, then sort the list
chars = sorted(list(set(text)))
# Assign characters to numbers
# enumerate makes an enumerate object with tuples of characters and their indices
# ex) list(enumerate("hello")) --> [(0, 'h'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')]
# dictionary = dict(keys, values)
# dict(letter or character, index) -----> {'a':0, 'b':1, 'c':2} as an example
char_indices = dict((c, i) for i, c in enumerate(chars))
# dict(index, letter or character) -----> {0:'a', 1:'b', 2:'c'} as an example
indices_char = dict((i, c) for i, c in enumerate(chars))

### Convert dataset into numbers
# convert chunks of data to numbers at a time
sentences = []
next_chars = []
average_sentence_length = 18
asl = average_sentence_length
# go through text data and chunk it up and enumerate things - can play around with numbers
# using asl to represent ~sentence length
# examples of i values in for loop: 0, 3, 6, 9, 12, ... number less than (len(text))-asl and is divisible by 3
# sentences list is a list of strings of traversing over all text by 3 indices each time
# next_chars list is a list of the subsequent letters after each of the strings in the sentences list
# do this so create a list of inputs and desired outputs (will convert these strings and characters into numbers for the network to train on)
### ex) text = 'This is a cool sentence and cheese is good'
# sentences = ['This is a cool sentence', 's is a cool sentence an', 's a cool sentence and c', ' cool sentence and chee']
# next_chars = [' ', 'd', 'h', 's']
for i in range(0, len(text) - asl, 3):
    sentences.append(text[i: i+asl])
    next_chars.append(text[i+asl])

### Creating x and y variables
# np.zeros(shape, datatype) where shape is making an array/matrix of specified dimensions
# datatype here is boolean so values are all False b/c np.zeros (later could turn to True) - np.zeros corresponds to False, np.ones corresponds to True
# x dimensions are (a,b,c) where a is the number of sentences(shifted by 3 to next index), b is the number of letters in each sentence, and c is the number of indices of characters
# In essence, saying x=inputs are a one-hot encoded list for every letter in the sentence
# and the y=outputs are the indices of the characters that follow each of the sentences in the x inputs 
# y dimensions are (a,c) because we are trying
# LSTM and RNN networks use bool to decide whether or not layer is active in training (relevant when dropout is used)
x = np.zeros((len(sentences), asl, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

### Convert letters to numbers (based on their indices in the indices_char dictionary)
# nested for loops - we are looping through the individual characters in the sentences list, converting letters to numbers 
# first for loop looping through all values in sentences (sentence = asl long string)
# second for loop looping through individual sentences (char = character in the sentence)
# i and sentence are counter variables
for i, sentence in enumerate(sentences):
    # t and char are counter variables
    for t, char in enumerate(sentence):
        # x inside second for loop because care about individual characters in the sentence 
        x[i, t, char_indices[char]] = 1
    # y outside second for loop because only care about what the sentence ends in, not the invidual characters in the sentence
    y[i, char_indices[next_chars[i]]] = 1



#=====Creating Neural Network=====#

### Create model
model = keras.models.Sequential()
# add first layer (LSTM model) with 128 neurons
# inputting 
model.add(keras.layers.LSTM(128, input_shape=(asl, len(chars))))# why chars?
# adding dense (fully connected) layer with (len(chars))# of neurons
# Remember: if don't know how many neurons want, use # of attributes
model.add(keras.layers.Dense(len(chars), activation='softmax'))

### Compile model
# categorical cross entropy means each data point only belongs to one category
# RMSprop good for this task
# learning rate (lr) low --> slower but get overall more optimized (weights for the) model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.005))

### Training model
# want epochs higher (~60)
model.fit(x, y, batch_size=19, epochs=57)



#=====Generating Original Text=====#

# model tries to predict which letter comes after a random asl character chunk in the text file
start_index = random.randint(0, len(text) - asl - 1)

# generated variable begins as small slice of text file and holds newly generated text
generated = text[start_index: start_index + asl]
generated += sentence

### 700 character long generated text
# doing ~same thing as outputting and training model as did with converting into numbers
for i in range(700):
    # similar to what I did above, here I create a non-boolean array (of 1s and 0s) to one-hot encode the randomly 
    # chosen starting text, and we put this into x_pred so we can use it in model.predict to output a predicted output value
    # The prediced output value is also a one-hot encoded list...? that we later break down in order to find the associated character with its index
    x_pred = np.zeros((1, asl, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1
    
    # predict next letter using model
    preds = model.predict(x_pred)[0]
    
    # instead of using a fxn to add randomness to output, I decided to simply take the argmax 
    # (which returns the index of the largest value in a list) of the predictions list to give the index
    # of the next character with the highest predicted probability
    next_index = np.argmax(np.asarray(preds))
    next_char = indices_char[next_index]

    # adding generated character to list of total generated characters
    generated += next_char
    # adding generated character to sentence list (add generated to end, remove first entry)
    sentence = sentence[1:] + [next_char]

### Saving generated text to text file
save_file_name = 'GeneratedSunTzu0.txt'
generated_string = ''.join(generated)
# writing generated text into text file
with open(save_file_name, 'w') as f:
    f.write(generated_string)
