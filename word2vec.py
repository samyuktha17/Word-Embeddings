import pandas as pd
import nltk
import gensim

df = pd.read_csv("C:\\Users\\Samuktha\\mshr\\test_word2vec\\jokes.csv")

x = df["Question"].values.tolist()
y = df["Answer"].values.tolist()

corpus = x + y

tok_corp= [nltk.word_tokenize(sentence) for sentence in corpus]

model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = 32)

model.save("testmodel")

model = gensim.models.Word2Vec.load("testmodel")

