#!/usr/bin/env python
# coding: utf-8

# In[214]:


import nltk
import re
import string
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt


# In[215]:


# reading the file, converted to lowercase
file = open("CBT_LM_Dataset/train.txt", "r", encoding = "utf8")
linesOG = []

for i in file:
    linesOG.append(i.replace('\n', '').lower())


# In[216]:


# For testing purposes use smaller dataset
lines = linesOG


# In[217]:


for i in range(len(lines)):
    lines[i] = lines[i].translate(str.maketrans('', '', string.punctuation))
    lines[i] = nltk.word_tokenize(lines[i])


# In[218]:


from collections import defaultdict
from nltk import bigrams

vocab = defaultdict(int)

for line in lines:
    for word in line:
        vocab[word] += 1

V = len(vocab)


# In[244]:


bicounts = defaultdict(lambda: defaultdict(int))
for line in lines:
    for w1, w2 in bigrams(line, pad_left=True, pad_right=True):
        bicounts[w1][w2] += 1


# ## No smoothing

# In[272]:


model1 = defaultdict(lambda: defaultdict(float))

for w1 in bicounts:
    total = sum(bicounts[w1].values())
    for w2 in bicounts[w1]:
        model1[w1][w2] = bicounts[w1][w2] / total


# ## Laplace smoothing

# In[273]:


model2 = defaultdict(lambda: defaultdict(float))

for w1 in bicounts:
    total = sum(bicounts[w1].values())
    for w2 in bicounts[w1]:
        model2[w1][w2] = (bicounts[w1][w2] + 1) / (total + V)


# ## Add k smoothing

# In[280]:


model3 = defaultdict(lambda: defaultdict(float))
k = 0.2

for w1 in bicounts:
    total = sum(bicounts[w1].values())
    for w2 in bicounts[w1]:
        model3[w1][w2] = (bicounts[w1][w2] + k) / (total + k*V)


# ## Testing the models

# In[281]:


q = []
o = []
a = []
output = []
with jsonlines.open('CBT_LM_Dataset/validation.jsonl') as f:
    for line in f.iter():
        output.append({"question": line["question"]})
        q.append(nltk.word_tokenize(line["question"].lower().translate(str.maketrans('', '', string.punctuation))))
        o.append(line["options"])
        a.append(line["answer"])


# In[282]:


def selectMax(model, prev, options):
    ans = None
    prob = -1
    for word in options:
        if(model[prev][word] > prob):
            ans = word
            prob = model[prev][word]
    return ans

def test(model):
    n = len(q)
    correct = [0]*n
    for i in range(n):
        prev = None
        for j in range(len(q[i])):
            if(q[i][j]=='xxxxx'):
                if(j!=0):
                    prev = q[i][j-1]
                break
        ans = selectMax(model, prev, o[i])
        output[i]["prediction"] = ans
        if(ans == a[i]):
            correct[i] = 1
    return correct


# ### Model 1: w/o smoothing

# In[283]:


correct = test(model1)
accuracy = sum(correct) / len(correct)
print('accuracy is', accuracy*100, '%')


# ### Model 2: Laplace smoothing

# In[284]:


correct = test(model2)
accuracy = sum(correct) / len(correct)
print('accuracy is', accuracy*100, '%')


# ### Model 3: Add k smoothing

# In[285]:


correct = test(model3)
accuracy = sum(correct) / len(correct)
print('accuracy is', accuracy*100, '%')


# ## Saving predictions to output file

# In[267]:


with jsonlines.open("output.jsonl", 'w') as f:
    for item in output:
        f.write(item)

