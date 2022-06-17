import nltk  
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer  
import json 
import pickle 
import random
import numpy as np  
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential  
from keras.layers import Dense, Activation, Dropout  
from keras.optimizers import * 
#from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()  

words=[]  
classes = []  
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()  
intents = json.loads(data_file)

for intent in intents['intents']:
    print("******* intent ******")
    print(intent)
    print("***********************")
    for pattern in intent['patterns']:   
        print("******* pattern ******")
        print(pattern)
        print("***********************")
        #tokenize each word   
        w = nltk.word_tokenize(pattern)   
        print("******* w tokenizada ******")
        print(w)
        print("***********************")
        words.extend(w)  
        print("******* words extend (w) ******")
        print(words)
        print("***********************")
        #add documents in the corpus   
        documents.append((w, intent['tag']))   
        print("******* documents ******")
        print(documents)
        print("***********************")
        # add to our classes list   
        if intent['tag'] not in classes:   
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates  
words = [lemmatizer.lemmatize(w.lower())  for w in words if w not in ignore_words]   
words = sorted(list(set(words)))
# sort classes  
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents  
print(len(documents), "documents")
# classes = intents  
print(len(classes), "classes", classes)
# words = all words, vocabulary  
print(len(words), "unique lemmatized words", words)  
pickle.dump(words,open('words.pkl','wb'))  
pickle.dump(classes,open('classes.pkl','wb'))


# create our training data  
training = []
# create an empty array for our output  
output_empty = [0] * len(classes)
print("****** output_empty ********** ", output_empty)
# training set, bag of words for each sentence 
for doc in documents:
    print("****** doc ********** ", doc)
    #initialize our bag of words   
    bag = []   
    # list of tokenized words for the pattern   
    pattern_words = doc[0]   
    print("****** pattern_words **********", pattern_words)
    # lemmatize each word - create base word, in attempt to represent related words   
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print("****** pattern_words 2 **********", pattern_words)
    # create our bag of words array with 1, if word match found in current pattern   
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)   
    print("***** BAG *******", bag)
    # output is a '0' for each tag and '1' for current tag (for each pattern)   
    output_row = list(output_empty)   
    print("**** output_row 1 ****", output_row)
    output_row[classes.index(doc[1])] = 1   
    print("****** output_row 2 ******", output_row)
    training.append([bag, output_row])
    print("LISTA 1: ", training)
    # shuffle our features and turn into np.array  
    random.shuffle(training)  
    print("LISTA 2: ", training)

print("LISTA 3: ", training)
training = np.array(training, dtype=object)
print("LISTA 4: ", training)
# create train and test lists. X - patterns, Y - intents  
train_x = list(training[:,0])  #columna 0
train_y = list(training[:,1])  # columna 1
print("Training data created")
print("***** trian x ******")
print(train_x)
print("----- train y ------")
print (train_y)

##########################################
################ modelo ##################
##########################################

# Create model - 3 layers. 
# First layer 128 neurons, 
# second layer 64 neurons and 
# 3rd output layer contains number of neurons  
# equal to number of intents to predict output intent with softmax  

model = Sequential()  
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model  

sgd =  keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model   

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)  
model.save('chatbot_model.h5', hist)  
print("model created")