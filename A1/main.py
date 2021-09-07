#!/usr/bin/env python
# coding: utf-8

# In[708]:


import nltk
import re
import string
import pandas as pd
import matplotlib.pyplot as plt


# In[709]:


# importing dataset
df = pd.read_csv('a01_spam.csv')
df.head()


# In[710]:


# info about the dataset
df.describe()


# In[711]:


#check for duplicates
dup = df[df.duplicated()]
dup.head()


# In[712]:


# removing duplicates
df.drop_duplicates(inplace=True)
df.describe()


# In[713]:


# check null values
df.isnull().sum()


# ### preprocessing complete, proceed with the tasks

# # Task 1: Counting words
# 
# #### Tokenization of words using Regex tokenizer
# 
# #### Punctuation and numbers are removed for better identification of actual words.

# In[714]:


# removing punctuation and numbers
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('[a-zA-Z]+')

df['t1'] = df.Message.apply(tokenizer.tokenize)
df.head()


# In[715]:


ccount = 0
vcount = 0
vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
for words in df['t1']:
    for word in words:
        if(word.startswith(vowels)):
            vcount +=  1
        else:
            ccount += 1

print('Total number of words starting with vowels:', vcount)
print('Total number of words starting with consonants:', ccount)


# # Task 2: Capitalised words
# #### We separate the data for ham and spam messages

# In[716]:


df_ham = df[df.Category == 'ham']
df_spam = df[df.Category == 'spam']
print('number of ham messages:', len(df_ham))
print('number of spam messages:', len(df_spam))


# In[717]:


# Checking ham messages
capcount = 0
count = 0

for words in df_ham['t1']:
    count += len(words)
    for word in words:
        if(word.isupper()):
            capcount +=  1

percent = capcount/count * 100
print('Percentage of capitalised word in ham messages: %f' % (percent), '%')


# In[718]:


# Checking spam messages
capcount = 0
count = 0

for words in df_spam['t1']:
    count += len(words)
    for word in words:
        if(word.isupper()):
            capcount +=  1

percent = capcount/count * 100
print('Percentage of capitalised word in spam messages: %f' % (percent), '%')


# # Task 3: Email IDs and Phone Numbers
# #### Check original messages for email IDs and phone numbers

# In[719]:


emails = {}
phone_nums = {}

mailregex = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
phoneregex = '[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'


# In[720]:


ham_mailcount = 0
ham_phonecount = 0

for message in df_ham['Message']:
    elist = re.findall(mailregex, message)
    plist = re.findall(phoneregex, message)
    if(len(elist)>0):
        ham_mailcount += 1
    if(len(plist)>0):
        ham_phonecount += 1

    # creating dictionary of emails and phone numbers
    for mail in elist:
        if(mail in emails):
            emails[mail] += 1
        else:
            emails[mail] = 1
    for pnum in plist:
        if(pnum in phone_nums):
            phone_nums[pnum] += 1
        else:
            phone_nums[pnum] = 1

epercent = ham_mailcount/len(df_ham) * 100
ppercent = ham_phonecount/len(df_ham) * 100
print('Percentage of ham messages containing emails: %f' % (epercent), '%')
print('Percentage of ham messages containing phone numbers: %f' % (ppercent), '%')


# In[721]:


spam_mailcount = 0
spam_phonecount = 0

for message in df_spam['Message']:
    elist = re.findall(mailregex, message)
    plist = re.findall(phoneregex, message)
    if(len(elist)>0):
        spam_mailcount += 1
    if(len(plist)>0):
        spam_phonecount += 1

    # creating dictionary of emails and phone numbers
    for mail in elist:
        if(mail in emails):
            emails[mail] += 1
        else:
            emails[mail] = 1
    for pnum in plist:
        if(pnum in phone_nums):
            phone_nums[pnum] += 1
        else:
            phone_nums[pnum] = 1

epercent = spam_mailcount/len(df_spam) * 100
ppercent = spam_phonecount/len(df_spam) * 100
print('Percentage of spam messages containing emails: %f' % (epercent), '%')
print('Percentage of spam messages containing phone numbers: %f' % (ppercent), '%')


# In[722]:


print('Total Number of emails found:', ham_mailcount+spam_mailcount)
print('In ham messages:', ham_mailcount)
print('In spam messages:', spam_mailcount)
print('Number of unique terms:', len(emails))
emails


# In[723]:


print('Total Number of phone numbers found:', ham_phonecount+spam_phonecount)
print('In ham messages:', ham_phonecount)
print('In spam messages:', spam_phonecount)
print('Number of unique terms:', len(phone_nums))
phone_nums


# In[724]:


totalmailmessage = spam_mailcount + ham_mailcount
totalphonemessage = spam_phonecount + ham_phonecount

sepercent = spam_mailcount / totalmailmessage * 100
sppercent = spam_phonecount / totalphonemessage * 100

print('Percentage of total messages with emails that are spam: %f' %(sepercent), '%')
print('Percentage of total messages with phone numbers that are spam: %f' %(sppercent), '%')


# In[725]:


hepercent = ham_mailcount / totalmailmessage * 100
hppercent = ham_phonecount / totalphonemessage * 100

print('Percentage of total messages with emails that are ham: %f' %(hepercent), '%')
print('Percentage of total messages with phone numbers that are ham: %f' %(hppercent), '%')


# # Task 4: Counting Currencies
# #### Check original messages for any currency symbols

# In[726]:


moneyregex = '([\£\$\€\¥]{1}[ ]*[,\d]+\.?\d*|[,\d]+\.?\d*[ ]*pounds?)'
spam_moneycount = 0
ham_moneycount = 0
mvalues = set()


# In[727]:


for message in df_ham['Message']:
    mlist = re.findall(moneyregex, message)
    if(len(mlist)>0):
        ham_moneycount += 1

    # creating set of monetary values
    for money in mlist:
        mvalues.add(money)

mpercent = ham_moneycount/len(df_ham) * 100
print('Percentage of ham messages containing monetary values: %f' % (mpercent), '%')


# In[728]:


for message in df_spam['Message']:
    mlist = re.findall(moneyregex, message)
    if(len(mlist)>0):
        spam_moneycount += 1

    # creating set of monetary values
    for money in mlist:
        mvalues.add(money)

mpercent = spam_moneycount/len(df_spam) * 100
print('Percentage of spam messages containing monetary values: %f' % (mpercent), '%')


# In[729]:


print('Total Number of monetary terms found:', ham_moneycount+spam_moneycount)
print('In ham messages:', ham_moneycount)
print('In spam messages:', spam_moneycount)
print('Number of unique terms:', len(mvalues))

mvalues


# # Task 5: Counting Emojis
# #### Check original messages for emoticons and print them

# In[730]:


from nltk.tokenize import TweetTokenizer
from nltk.tokenize.casual import EMOTICON_RE

emoticons = set()
tk = TweetTokenizer()
df['t5'] = df.Message.apply(tk.tokenize)
df.head()


# In[731]:


count = 0
for words in df['t5']:
    smileys = EMOTICON_RE.findall(' '.join(words))
    for emoti in smileys:
        emoticons.add(emoti)
        count += 1


# In[732]:


print('Total number of emoticons:', count)
print('Unique emoticons:', len(emoticons))
emoticons


# # Task 6: Counting Clitics
# #### Check original messages for words with clitics and print them

# In[733]:


s = 'sd ds dad pois'
x = re.search('dad\s', s)
x.group()


# In[734]:


cliticregex = '[a-zA-Z]+\'[a-zA-Z]{1,2}$'
clitic_words = set()


# In[735]:


for words in df['t5']:
    for word in words:
        cword = re.search(cliticregex, word)
        if(cword):
            clitic_words.add(cword.group())


# In[736]:


print('number of words with clitics:', len(clitic_words))
clitic_words


# # Task 7: Starting with
# #### Check original messages and print those that start with a given word

# In[737]:


word = input('Enter input word: ')
regex = '^'+word+'\s'


# In[738]:


count = 0
for message in df['Message']:
    match = re.search(regex, message)
    if(match):
        print(message)
        count += 1


# In[739]:


print('Number of messages starting with the word \'%s\' are - %d' %(word, count))


# # Task 8: Ending with
# #### Check original messages and print those that end with a given word

# In[740]:


word = input('Enter input word: ')
regex = '\s'+word+'$'


# In[741]:


count = 0
for message in df['Message']:
    match = re.search(regex, message)
    if(match):
        print(message)
        count += 1


# In[742]:


print('Number of messages ending with the word \'%s\' are - %d' %(word, count))


# # Task 9: Classification using heuristics
# #### Check original messages and classify them using various heuristics

# #### Only 2
# Using the observations from task 2, we classify a message as spam if more than 20% of the words in the message are capitalized.
# This is done because ham messages also have the likelihood of containing some capitalized words.

# In[743]:


def isSpamCapital(words):
    capcount = 0
    count = len(words)
    if(count==0):
        return 'ham'
    for word in words:
        if(word.isupper()):
            capcount +=  1
    percent = capcount / count * 100
    if(percent > 20):
        return 'spam'
    else:
        return 'ham'

only2 = df['t1'].apply(isSpamCapital)


# In[744]:


correct = (df['Category']==only2).sum()
percent = correct / len(df) * 100
print('accuracy of \"only 2\" heuristic is:', percent)


# #### Only 3
# Using the observations from task 3, we can see that spam messages are very likely to contain emails and phone numbers, while ham messages do not. So if a message contains either, we classify it as spam

# In[745]:


mailregex = '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
phoneregex = '[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'

def isSpamEP(message):
    mmatch = re.search(mailregex, message)
    pmatch = re.search(phoneregex, message)
    if(mmatch or pmatch):
        return 'spam'
    else:
        return 'ham'

only3 = df['Message'].apply(isSpamEP)


# In[746]:


correct = (df['Category']==only3).sum()
percent = correct / len(df) * 100
print('accuracy of \"only 3\" heuristic is:', percent)


# #### Only 4
# Using the observations from task 4, we can see that spam messages have a much higher chance to contain monetary amounts, while ham messages do not. So if a message contains a monetary quantity we classify it as spam

# In[747]:


moneyregex = '([\£\$\€\¥]{1}[ ]*[,\d]+\.?\d*|[,\d]+\.?\d*[ ]*pounds?)'

def isSpamMoney(message):
    match = re.search(moneyregex, message)
    if(match):
        return 'spam'
    else:
        return 'ham'

only4 = df['Message'].apply(isSpamMoney)


# In[748]:


correct = (df['Category']==only4).sum()
percent = correct / len(df) * 100
print('accuracy of \"only 4\" heuristic is:', percent)


# #### both23
# We classify a message as spam if it satisfies either heuristic 2 or heuristic 3

# In[749]:


both23 = ['spam' if (v1=='spam' or v2=='spam') else 'ham' for v1, v2 in zip(only2, only3)]


# In[750]:


correct = (df['Category']==both23).sum()
percent = correct / len(df) * 100
print('accuracy of \"both 2 and 3\" heuristic is:', percent)


# #### both34
# We classify a message as spam if it satisfies either heuristic 3 or heuristic 4

# In[751]:


both34 = ['spam' if (v1=='spam' or v2=='spam') else 'ham' for v1, v2 in zip(only3, only4)]


# In[752]:


correct = (df['Category']==both34).sum()
percent = correct / len(df) * 100
print('accuracy of \"both 3 and 4\" heuristic is:', percent)


# #### both24
# We classify a message as spam if it satisfies either heuristic 2 or heuristic 4

# In[753]:


both24 = ['spam' if (v1=='spam' or v2=='spam') else 'ham' for v1, v2 in zip(only2, only4)]


# In[754]:


correct = (df['Category']==both24).sum()
percent = correct / len(df) * 100
print('accuracy of \"both 2 and 4\" heuristic is:', percent)


# #### all234
# We classify a message as spam if it satisfies either heuristic 2 or heuristic 3 or heuristic 4

# In[755]:


all234 = ['spam' if (v1=='spam' or v2=='spam' or v3=='spam') else 'ham' for v1, v2, v3 in zip(only2, only3, only4)]


# In[756]:


correct = (df['Category']==all234).sum()
percent = correct / len(df) * 100
print('accuracy of \"all 2,3,4\" heuristic is:', percent)


# ## From the above results we can notice that heuristic 3 is the most accurate at predicting spam, followed by heuristic 4 then heuristic 2
