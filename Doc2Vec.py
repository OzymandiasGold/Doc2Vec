# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the restaurant dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    

# Tokenization and tagging of the data.
    
nltk.download('punkt')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


data = corpus
tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i,_d in enumerate(data)]



# Training the actual model.

max_epochs = 100
vec_size = 90 # Trying something new here, generally speaking the more dimansions the better the outcome.
              # As there are more area for similarity to be found in.
alpha = 0.025


model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)


for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # Decreasing the learning rate.
    model.alpha -= 0.0002
    
    # Fix the learning rate , no decay.
    model.min_alpha = model.alpha
    
model.save("d2v.model")
print("Model Saved")



# Displaying the actual Vectors of different documents.
# Currently there are 3 different descriptions. 
# Two are similar and one is different from both.

test_data = word_tokenize("Cosy, Warm, Family.".lower())
v1 = model.infer_vector(test_data)

test_data2 = word_tokenize('''I moved into the area last year and I'm so glad this place is on my doorstep. Not only are their pizzas fresh,
                           original, and extremely tasty, but the service is second to none. The owners are lovely, and every time I go in they are friendly and polite. It's a gem of a restaurant in this area. I’m just glad it’s within walking distance!
They do takeaway also and a particular favourite of mine is the margarita, simple but tasty. 

Unfortunately, every time I’ve been here they have run out of deserts so I’m looking forward to trying one eventually as they look delightful'''.lower())
v2 = model.infer_vector(test_data2)

test_data3 = word_tokenize('''Always something wrong with my order whenever I go. Either something forgotten or swapped. Last time instead of a chicken mayo they gave me a cheeseburger. They forget my fries or my apple pie. The food never looks presentable - the burgers all over the place with the ketchup dripping down, bun sliding off. 
Also which other McDonald’s you know needs a security guard...  
Toilets filthy, soap never in the container. 
Honestly you can do better than this McDonald’s tooting. Not impressed'''.lower())
v3 = model.infer_vector(test_data3)


# Reshaping these vectors and showing their cosine similarity.

from sklearn.metrics.pairwise import cosine_similarity

v1 = v1.reshape(1,-1)
v2 = v2.reshape(1,-1)
v3 = v3.reshape(1,-1)

result1 = cosine_similarity(v1,v3)
result2 = cosine_similarity(v1,v2)

if (result1>result2):
    print('bad')
else:
    print('good')
    
    
print(result1)
print(result2)

#print(cosine_similarity(v1,v3, ' This should be small.'))
#print(cosine_similarity(v1,v3, ' This should be high.'))
#print(cosine_similarity(v2,v3, ' This should be medium.'))


#print(cosine_similarity(v1,v2, ' This should be large'))










    
