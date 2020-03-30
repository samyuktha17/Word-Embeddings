import pandas as pd
import nltk
import gensim

df = pd.read_csv("file_path\jokes.csv") # load the file into a dataframe

x = df["Question"].values.tolist() 
y = df["Answer"].values.tolist()

corpus = x + y # create a corpus

tok_corp= [nltk.word_tokenize(sentence) for sentence in corpus] # tokenize the corpus

model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = 32) # pass the tokenized corpus to the model ; consider every word that appears >=1 time ; size of the vector = 32

model.save("testmodel")

model = gensim.models.Word2Vec.load("testmodel")

boys = model.most_similar("Boy") # find words that are associated with the word "Boy" 
girls = model.most_similar("Girl") # find words that are associated with the word "Girl"

print(boys)
print(girls)
