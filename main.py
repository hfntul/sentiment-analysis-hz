import pandas as pd                                                   # dataframe
import snscrape.modules.twitter as sntwitter                          # crawling twitter data
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # sentiment score
import preprocessor as p                                              # clean data
from wordcloud import WordCloud, STOPWORDS                            # create wordcloud
import matplotlib.pyplot as plt                                       # plotting wordcloud
import seaborn as sns                                                 # plotting heatmap
import numpy as np                                                    # for image processing
from PIL import Image                                                 # for image processing

#############################
###### Some may not in ######
#### chronological order ####
#############################

###### SCRAPPING TWEET ######

# the scraped tweets, this is a generator
scraped_tweets = sntwitter.TwitterSearchScraper('zemo since:2021-04-09 until:2021-04-10').get_items()

# convert to a DataFrame and keep only relevant columns
df = pd.DataFrame(scraped_tweets)[['id', 'user', 'date', 'content', 'lang', 'retweetCount', 'likeCount', 'quoteCount']]

###### PRE-PROCESSING ######

# lower case
df['content'] = df['content'].str.lower()

# remove url mention and emoji
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
df['content'] = df['content'].apply(p.clean)

# replace dirty words
df['content'] = df['content'].str.replace("<<dirty words>>", "****")

# replace symbol
df['content'] = df['content'].str.replace("&amp;", "&").str.replace("&gt;", ">").str.replace("&lt;", "<")

# pick tweets
temp = df[df['content'].str.contains('zemo')==True]

# remove non-english tweet
temp = temp[temp['lang'].str.contains('en')==True]

# remove duplicate
temp.drop_duplicates(subset = 'content',inplace = True)

# remove hashtag
p.set_options(p.OPT.HASHTAG)
temp['content'] = temp['content'].apply(p.clean)

# put only username
temp['user'] = temp['user'].str.replace("{'username': |'|,", "")
temp['user'] = temp['user'].str.split(' ').str[0]

# get only 30 tweets per user
fetch = temp.drop_duplicates(['user', 'content']).sort_values(by='likeCount', ascending=False).groupby('user').head(30)
data_clean = temp.merge(fetch)

###### MOST ACTIVE USER ###### (not in order)

user_count = data_clean['user'].value_counts()[:10]
plt.figure(figsize=(20,10))
sns.barplot(user_count.index, user_count.values, alpha=0.8)
plt.title('Most Active User',fontsize=16 )
plt.ylabel('Amount of Tweets', fontsize=14)
plt.xlabel('User', fontsize=14)
plt.show()

###### CREATING WORDCLOUND ###### (not in order)

# set stopwords
stopwords = set(STOPWORDS)
stopwords.update(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                  'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
                  'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
                  'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
                  'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                  'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
                  'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                  'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
                  'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                  'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                  'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                  'won', "won't", 'wouldn', "wouldn't", 'doesnt', 'ok', 'okay', 'really', 'got', 
                  'im', 'cant','baron','helmut', 'two', 'dont', 'hes', 'even', 'gonna', 'getting',
                  'actually','yeah', 'ye', 'let', 'literally', 'oh', 'going', 'didnt', 'maybe', 'us',
                  'well', 'yes', 'lol','episode','spoiler', 'spoilers', 'thing', 'see', 'think', 'know', 
                  'still', 'go', 'say', 'said', 'wait', 'want','one','watching','whole', 'watch', 'ep',
                  'though','thats','lot','thought']) 

# plotting
wordcloud = WordCloud(width=800, colormap='Set2',stopwords=stopwords,height=400,max_font_size=200, max_words = 100, collocations=False, background_color='black').generate(' '.join(data_clean['content']))
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# plotting using image
mask = np.array(Image.open('/content/drive/MyDrive/Twitter/mask.jpg'))
wordcloud = WordCloud(width=1600, mask = mask,stopwords=stopwords,height=400,max_font_size=200, max_words = 100, collocations=False, background_color='black').generate(' '.join(data_clean['content']))
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###### SENTIMENT ANALYSIS ######

# gen sentiment scores
analyzer  = SentimentIntensityAnalyzer()
sentiment = data_clean['content'].apply(lambda x: analyzer.polarity_scores(x))

# adding sentiment into dataframe
dataScore = pd.concat([data_clean, sentiment.apply(pd.Series)],1)

# adding the overall sentiment
dataScore.loc[dataScore['compound'] >  -0.05, 'overall'] = 'neutral' 
dataScore.loc[dataScore['compound'] >= 0.05, 'overall'] = 'positive' 
dataScore.loc[dataScore['compound'] <= -0.05, 'overall'] = 'negative' 

# plotting 
dataScore['overall'].value_counts().sort_values(ascending=False)
score_count = dataScore['overall'].value_counts()
plt.figure(figsize=(15,10))
colors = sns.color_palette('pastel')[0:5]
plt.title('Overall Sentiment',fontsize=20)
plt.pie(score_count.values, colors = colors, labels = score_count.index, autopct='%.0f%%', textprops={'fontsize': 16})
plt.show()

# historgram all tweets
dataScore['compound'].hist()

# histogram sample
dataScore_nz = dataScore[dataScore['compound'] != 0]
dataScore_nz['compound'].sample(5000).hist()

###### TWEET EXAMPLE ######

# positive
j = 0
for i in dataScore[dataScore['compound'] >= .9]['content'].sample(3):
    j = j + 1
    print(str(j) + '. ' + i)
    print(' ')

# negative
j = 0
for i in dataScore[dataScore['compound'] <= -.9]['content'].sample(3):
    j = j + 1
    print(str(j) + '. ' + i)
    print(' ')

###### EVALUATING ######

import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# convert 'content' to the correct format
def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()
data = [preprocess_text(t) for t in dataScore['content']]

# pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

# split data test data train
x_train, x_test, y_train, y_test = train_test_split(data, dataScore['overall'], test_size=0.2, random_state=42)

# classification_report
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='f1_macro')
clf.fit(x_train, y_train)
print(classification_report(y_test, clf.predict(x_test), digits=4))

# heat map confusion matrix
sns.heatmap(confusion_matrix(y_test, clf.predict(x_test)), annot = True, fmt='g')