import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering

sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
    "She enjoys reading books",
    "He is walking his dog",
    "The weather is nice today",
    "Let's go to the beach",
    "I need a cup of coffee",
    "They are planning a trip",
    "She cooks delicious meals",
    "I am learning to play the piano",
    "He works at a tech company",
    "This book is very interesting",
    "The sun is shining brightly",
    "It's raining outside",
    "The children are playing soccer",
    "He is fixing the car",
    "She writes beautiful poetry",
    "The flowers are blooming",
    "I enjoy hiking in the mountains",
    "He is swimming in the pool",
    "We are watching a comedy show",
    "The dog is barking loudly",
    "She is painting a landscape",
    "He likes to play video games",
    "I am studying for my exams",
    "They are building a new house",
    "The coffee shop is busy",
    "He listens to classical music",
    "She is designing a website",
    "I bought a new laptop",
    "They are baking cookies",
    "The bird sings in the morning",
    "He loves going to concerts",
    "She practices yoga every day",
    "The phone is ringing",
    "I need to finish this project",
    "The restaurant has great food",
    "He collects rare coins",
    "She enjoys watching documentaries",
    "They are exploring the city",
    "The laptop is charging",
    "He is training for a marathon",
    "I meditate every morning",
    "The bakery sells fresh bread",
    "She is learning French"
]

from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer('all-MiniLM-L6-v2')

paraphrases = paraphrase_mining(model, sentences)
paraphrases[0:20]


for paraphrase in paraphrases[0:20]:
    score, i, j = paraphrase
    print("{} \t\t\t\t\t {} \t\t\t Score: {:.4f}".format(sentences[i], sentences[j], score))