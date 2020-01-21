#%% 
# LOAD DATA 
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from functools import partial
import itertools
from itertools import chain
from collections import Counter
import textblob
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer 
from pandas import read_excel

df = pd.read_csv(r"emotiondata_cleaned.csv")

#%% 
#GET POS-TAGGING
#define tokenizer and POS-tagger
tok_and_tag = lambda x: pos_tag(word_tokenize(x))

#adding the POS-tagging
#df['sentence'] = df['sentence'].apply(tok_and_tag)
# to lower all words
df['lower_sent'] = df['sentence_ut'].apply(str.lower)
#df['lower_sent'].apply(tok_and_tag)
df['tagged_sent'] = df['lower_sent'].apply(tok_and_tag)


tokens, tags = zip(*chain(*df['tagged_sent'].tolist()))
tags
possible_tags = sorted(set(tags))
possible_tags_counter = Counter({p:0 for p in possible_tags})

#iterating through each sentence
df['pos_counts'] = df['tagged_sent'].apply(lambda x: Counter(list(zip(*x))[1]))
df['pos_counts']

#add in the POS that don't appears in the sentence with 0 counts:

def add_pos_with_zero_counts(counter, keys_to_add):
     for k in keys_to_add:
         counter[k] = counter.get(k, 0)
     return counter

df['pos_counts_with_zero'] = df['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))

#flattening the values to a list
df['sent_vector'] = df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])

#now creating new matrix dataframe with values
df2 = pd.DataFrame(df['sent_vector'])

df3 = pd.DataFrame(df2['sent_vector'].values.tolist(), columns=['$','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','MD','NN','NNP','NNS','PDT','PRP','PRP$','RB','RBR','RBS','RP','TO','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB'])
del df3['$']

#%% 
# ADD COLUMNS TO MAKE MORE FEATURES
df3['label'] = df['emotions']
df3['sentence_tokenize'] = df['sentence']
df3['sentence_lower'] = df['sentence_ut']

#%%
# GET WORD COUNT
df3['word_count'] = df3['sentence_lower'].str.split().str.len()

#%%
# GET POLARITY AND SUBJECTIVITY
df3['polarity'] = df3['sentence_lower'].map(lambda text: TextBlob(text).sentiment.polarity)
df3['subjectivity'] = df3['sentence_lower'].map(lambda text: TextBlob(text).sentiment.subjectivity)

#%%
# GET CONCRETENESS SCORE
lemmatizer = WordNetLemmatizer() 
conc_database = pd.read_excel('Concreteness_ratings.xlsx', encoding='latin-1')

def get_concreteness(text):
    word_list = text.split(' ')
    word_list = list(lemmatizer.lemmatize(word) for word in word_list)
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    concreteness_score = df_with_concr['Conc.M'].mean()
    return concreteness_score
df3['concreteness'] = df3['sentence_lower']. apply(get_concreteness)


#%% 
#WRITE TO CSV
df3.to_csv('features.csv', index = None, header=True)


