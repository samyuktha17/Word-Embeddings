import pandas as pd
import nltk
import gensim

df = pd.read_csv("file_path\jokes.csv")

x = df["Question"].values.tolist()
y = df["Answer"].values.tolist()

corpus = x + y

tok_corp= [nltk.word_tokenize(sentence) for sentence in corpus]

model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = 32)

model.save("testmodel")

model = gensim.models.Word2Vec.load("testmodel")

boys = model.most_similar("Boy")
girls = model.most_similar("Girl")

print(boys)
print(girls)
