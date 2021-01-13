```python
%qtconsole
```

<H4>Some basic points about NLP</H4>
<br>
Natural Language Processing or NLP is basically a sub-field of artificial intelligence in which we make computer system learn, analyse and generate natural language
<hr style="height: 2pt;background-color: red">
NLP : consists of NLU and NLG<br>
       NLU - Natural Language Understanding
       NLG - Natural Language Generation
<h5>5 different phases of NLU and NLG</h5>
<ol>
<li>Lexical Processing:- tokenisation. morphological analysis, processing on individual words</li>
<li>Syntactic Processing :- Internal representation of the text, example a parse tree representation.</li>
<li>Semantic Processing :- Clarifying the meaning of the word, meaning of words may be different in different context, for example, Federal Bank, bank of a river</li>
<li>Disposal/Pragmatic Processing:- Former deals with emotions (like text to speech) and Pragmatic deals with stories (eg John is a monk. He goes to Church Daily. He is a Catholic.)</li>
</ol>
<hr style="height: 2pt;background-color: blue">

<h1>Text Summmarisation System</h1>
<hr style="height: 2pt;background-color: green">
Condensing a longer document into a short concise document without losing the core information
<br>
Based on input, it can be a sinlge document or multi-document summary
<br>Based on the Purpose: Like some documents are generic or some from one domain (like summarising covid-19 dataset is domain)
<br>Query Based: User asks questions.
<h6>Extractive (just retains main sentences) and Abstractive (writing the summary in own words)</h6>


<It's assumed you are familiar with supervised and unsupervised learning>
<hr style="height: 2pt;background-color: red">

<h4>Text summariation by taking into account various features</h4>
<br>It involves the following steps

<ol>
    <li>Pre Processing
        <ul>
            <li>Sentence Segmentation</li>
            <li>Tokenization</li>
            <li>Stop-Words Removal</li>
            <li>Stemming</li>
            </ul>
        </li>
    <li>Feature Extraction
        <ul>
            <li>Word Score</li>
            <li>Sentence Score</li>
        </ul>
 </ol>
 
 <h6>Quotes are an important part of summary </h6>


```python
#Importing the Libraries
#NLTK-natural language toolkit for natural language processing
#CORPUS- Collection of Documents, eg Wall Street Journal CORPUS
#using stop-words CORPUS, stop-words are words like of, are, is etc, 
#which occur more frequently and have no semantic meaning
#We need to tokenize the words because we need to compute the frequency of each word
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
#Stemmer goes
```


```python
#import documents
f = open(('./trial_covid_dataset.txt'),"r")
text = f.read()
f.close()
```


```python
#So, we have stored the document's text into text variable
#Preprocessing the data : Very Important to avoid overfit or underfit

#Step-1 We tokenize each sentence

sent_tokens = nltk.sent_tokenize(text)
word_tokens = nltk.word_tokenize(text)

#Step-2 We convert to lower case
word_tokens_lower = [word.lower() for word in word_tokens]

#Step-3 remove stopwords
stopWords = list(set(stopwords.words('english'))) #getting all stopwords of English and storing in StopWords
word_tokens_refined = [word for word in word_tokens_lower if word not in stopWords]
```


```python
ps = PorterStemmer()
stem = []
for word in word_tokens_refined:
    stem.append(ps.stem(word))
    #storing all the variants of the word
word_tokens_refined=stem
```

<b>The goal of a stem is to remove as much variance as possible so as to fit in different cases</b>


```python
# #It hasn't been run yet
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# stem = []
# wnl = WordNetLemmatizer()
# for word in word_tokens_refined:
#     if wnl.lemmatize(word).endswith('e'):
#         stem.append(wnl.lemmatize(word))
#     else:
#         stem.append(ps.stem(word))
#     #storing all the variants of the word
# word_tokens_refined=stem
```

<h4>Finding the number of Proper Nouns in each sentence</h4>


```python
#proper noun
proper_noun = {} #empty dict
from nltk.tag import pos_tag #part of speech tag
#for each sentence we tag each sentence
for sentence in sent_tokens:
    proper_noun[sentence]=0 #initialising the dictionary value to zero
    tagged_sentence=pos_tag(sentence.split()) #getting a tagged_sentence list for tagging the words
    #in this sentence
    proper_nouns_in_this_sentence = [word for word,pos in tagged_sentence if pos=="NNP"] #you know how to write this one-liner in multiple ways
    proper_noun[sentence]=len(proper_nouns_in_this_sentence)
#So in proper noun,we get the score of each sentence based on proper nouns
```

<h5>Normalising the number of proper nouns in each sentence</h5>


```python
#Normalise the proper_noun DICTIONARY
maximum = max(proper_noun.values())
for key in proper_noun:
    try:
        proper_noun[key] = proper_noun[key]/maximum
    except ZeroDivisionError:
        x = 0
```

<h4>Based on Number of Cue Phrases in Each Sentence</h4>
<hr>
<h6>Getting the list of cue_phrases </h6>


```python
#Cue Phrases - Highly important words and signify the importance of a sentence, signify is also a cue-phrase
#Getting the *list* of qphrases
qphrases = []
import pandas as pd
import numpy as np
a = pd.read_csv('./cue_phrases.csv')
#ideally get the list from tech retireval conference society
#This list is incomplete. it won't yield a good summary
b = np.array(a)
for i in range(0,len(b)):
    qphrases.append(b[i][0])
```

<h4>Finding the Number of Cue Phrases in Each Sentence</h4>


```python
cue_phrase_dict = {}
for sentence in sent_tokens:
    cue_phrase_dict[sentence]=0
    word_tokens_in_this_sentence=nltk.word_tokenize(sentence)
    for word in word_tokens_in_this_sentence:
        if word.lower() in qphrases:
            cue_phrase_dict[sentence]+=1
```

<h5>Normalising the number of cue_phrases in each sentence</h5>


```python
#normaalising the values of cue_words
maximum = max(cue_phrase_dict.values())
for key in cue_phrase_dict:
    try:
        cue_phrase_dict[key] = cue_phrase_dict[key]/maximum
    except ZeroDivisionError:
        x = 0
```

<B>So far, we have discussed proper nouns and cue_words. Now coming to numerical digits


```python
numeric_data = {}
for sentence in sent_tokens:
    numeric_data[sentence]=0
    word_tokens_in_this_sentence = nltk.word_tokenize(sentence)
    for word in word_tokens_in_this_sentence:
        if word.isdigit():
            numeric_data[sentence]+=1
```

<h5>Normalising the number of numeric data in each sentence</h5>


```python
maximum = max(numeric_data.values())
for key in numeric_data:
    try:
        numeric_data[key] = numeric_data[key]/maximum
    except ZeroDivisionError:
        x = 0
```

<b>Similarly, we can find uppercase, formatting, like word.isdigit()
#Word_Frequency_can also be added! Try it once
<br>
    <hr style="height: 2px;">
<b><u>Similarly, in the following code bloks, we take into account sentence length, sentence ordering, uppercase sentence,heading matches, frequency</u> </b>
    
<hr style="height: 2px">

<h5>Formula for sentence length</h5>

![sent_length](https://user-images.githubusercontent.com/75802971/104480746-e6557300-55ea-11eb-8dfc-8a4f9e8b7d22.jpg)


```python
#sentence_length
sentence_length = {}
for sentence in sent_tokens:
    word_tokens_in_this_sentence=nltk.word_tokenize(sentence)
    if len(word_tokens_in_this_sentence) in range(0,10):
        sentence_length[sentence]=1-0.058*(10-len(word_tokens_in_this_sentence))
    elif len(word_tokens_in_this_sentence) in range(10,20):
        sentence_length[sentence]=1
    else:
        sentence_length[sentence] = 1-0.05*(len(word_tokens_in_this_sentence)-20)
```

<hr style="height: 2px">
<h5>Formula for sentence Position</h5>
<img src="./sent_pos.jpg">


```python
#sentence_position
sentence_position={}
n = 1
N = len(sent_tokens)
for sentence in sent_tokens:
    a =1/n
    b = 1/(N-n+1)
    sentence_position[sentence]=max(a,b)
    n = n+1
```


```python
#Word matches with Heading
head_match = {} #empty dictionary
heading=sent_tokens[0] #first sentence as the heading

#Now for heading matching

for sentence in sent_tokens:
    head_match[sentence]=0 #intitally heading match with that sentence is zero
    word_tokens_in_this_sentence = nltk.word_tokenize(sentence)
    for word in word_tokens_in_this_sentence:
        if word not in stopWords:#if the word is not a stopword
            word = ps.stem(word) #stemming the word
            if word in ps.stem(heading):#check if its in heading
                head_match[sentence]+=1
```


```python
#Normalising the heading matches
maximum = max(head_match.values())
for key in head_match:
    try:
       head_match[key] = head_match[key]/maximum
    except ZeroDivisionError:
        x = 0
```


```python
upper_case={}
for sentence in sent_tokens:
    upper_case[sentence] = 0
    word_tokens_in_this_sentence = nltk.word_tokenize(sentence)
    for word in word_tokens_in_this_sentence:
        if word.isupper():
            upper_case[sentence] += 1
```


```python
#normalising the upper case dictionary values

maximum = max(upper_case.values())
for key in head_match:
    try:
       upper_case[key] = head_match[key]/maximum
    except ZeroDivisionError:
        x = 0
```


```python
import math
#frequency table
freqTable = {}
for word in word_tokens_refined:    
    if word in freqTable:         
        freqTable[word] += 1    
    else:         
        freqTable[word] = 1
        

for k in freqTable.keys():
    freqTable[k]= math.log10(1+freqTable[k])
    

#computing word frequency for each sentence
word_frequency={} #empty dictionary for word_frequency
for sentence in sent_tokens:
    word_frequency[sentence]=0
    unstemmed_word_tokens_in_this_sentence=nltk.word_tokenize(sentence)
    stemmed_Word_tokens_in_this_sentence=[]
    for word in unstemmed_word_tokens_in_this_sentence:
        stemmed_Word_tokens_in_this_sentence.append(ps.stem(word))
        #so we have got stemmed words for this sentence
    for word,freq in freqTable.items():
        if word in stemmed_Word_tokens_in_this_sentence:
            #if thw word is in the frequency table and in stemmd word tokens of this sentence, we add
            #the frequency of the word to this sentence
            word_frequency[sentence]+=freq
```


```python
#normalising the word_frequency

maximum = max(word_frequency.values())
for key in head_match:
    try:
       word_frequency[key] = word_frequency[key]/maximum
    except ZeroDivisionError:
        x = 0
```


```python
total_score={}
for k in sent_tokens:
    total_score[k]=cue_phrase_dict[k]+numeric_data[k]+sentence_length[k]+sentence_position[k]+word_frequency[k]+upper_case[k]+proper_noun[k]+head_match[k]
```


```python
#Now retaining the important information using average
import numpy as np
average = np.mean(list(total_score.values()))

summary = ''

for sentence in sent_tokens:
    if total_score[sentence]>average:
        summary = summary+sentence
print(summary)
```

    Success from two leading coronavirus vaccine programs likely means other frontrunners will also show strong protection against COVID-19, Bill Gates said Tuesday.The fact that two coronavirus vaccines recently showed strong protection against COVID-19 bodes well for other leading programs led by AstraZeneca, Novavax, and Johnson & Johnson, Bill Gates said Tuesday.The billionaire Microsoft founder and philanthropist said it will be easier to boost manufacturing and distribute these other shots to the entire world, particularly developing nations.The vaccine space has seen a flurry of good news in recent days, marked by overwhelming success in late-stage trials by both Pfizer and Moderna."With the very good news from Pfizer and Moderna, we think it's now likely that AstraZeneca, Novavax, and Johnson & Johnson will also likely show very strong efficacy," Gates told journalist Andrew Ross Sorkin.The scientific success has turned the top challenges surrounding a COVID-19 vaccine to the manufacturing and distribution front.Gates noted that the world will be supply constrained for 2021, but these additional vaccines will prove valuable on that front.
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
