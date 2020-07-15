import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import StratifiedKFold
import os

# pytorch
import torch
import torch.optim as optim

import random 

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformers
from transformers import *
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

# tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K

import tokenizers
import pickle
import math
import re
import string
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
from nltk import bigrams
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')
import collections
from wordcloud import WordCloud
from textwrap import wrap
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener
import tweepy
import csv
import time
import datetime

from flask import Flask, request, jsonify, make_response
from flask import render_template, url_for, flash, redirect, copy_current_request_context
from flask_socketio import SocketIO, emit
from threading import Thread, Event

from flask_restful import reqparse, abort, Api, Resource
import json
from flask import jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba012'
app.config['DEBUG'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

# Twitter credentials
consumer_key = 'Your key'
consumer_secret = 'Consumer secret goes here'
access_key = 'Your access key'
access_secret = 'Your access secret'

# Pass your twitter credentials to tweepy via its OAuthHandler
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
t_api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits

def predict_sentiment(learner, text):
  sentiment = learner.predict(text)[1].item()
  return sentiment

def sentiment_label (Sentiment):
   if Sentiment == 2:
       return "positive"
   elif Sentiment == 0:
       return "negative"
   else :
       return "neutral"

def replace_url(string): # cleaning of URL
    text = re.sub(r'http\S+', 'LINK', string)
    return text


def replace_email(text):#Cleaning of Email related text
    line = re.sub(r'[\w\.-]+@[\w\.-]+','MAIL',str(text))
    return "".join(line)

def rep(text):#cleaning of non standard words
    grp = text.group(0)
    if len(grp) > 3:
        return grp[0:2]
    else:
        return grp# can change the value here on repetition
def unique_char(rep,sentence):
    convert = re.sub(r'(\w)\1+', rep, sentence) 
    return convert

def find_dollar(text):#Finding the dollar sign in the text
    line=re.sub(r'\$\d+(?:\.\d+)?','PRICE',text)
    return "".join(line)

def replace_emoji(text):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F" # emoticons
    u"\U0001F300-\U0001F5FF" # symbols & pictographs
    u"\U0001F680-\U0001F6FF" # transport & map symbols
    u"\U0001F1E0-\U0001F1FF" # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text) 

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

def clean_text(text: str) -> str:
    text = str(text)
    for punct in puncts + list(string.punctuation):
        if punct in text:
            text = text.replace(punct, f'')
    return text
   
def replace_asterisk(text):
    text = re.sub("\*", 'ABUSE ', text)
    return text

def remove_duplicates(text):
    text = re.sub(r'\b(\w+\s*)\1{1,}', '\\1', text)
    return text

def change(text):
    if(text == ''):
        return text
  #calling the subfunctions in the cleaning function
    text = replace_email(text)
    text = replace_url(text)
    text = unique_char(rep,text)
    text = replace_asterisk(text)
    text = remove_duplicates(text)
    text = clean_text(text)
    return text

def extract_tweets(search_words,date_since, date_until, numTweets):
  return(tweepy.Cursor(t_api.search, q=search_words, lang="en", since=date_since, until=date_until, tweet_mode='extended').items(numTweets))

def scraptweets(search_words, date_since, date_until, numTweets, numRuns):
    # Define a pandas dataframe to store the date:
    db_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'following', 'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount', 'text', 'hashtags'])
    db_tweets['hashtags'] = db_tweets['hashtags'].astype('object')
    #db_tweets = pd.DataFrame()

    for i in range(numRuns):

        tweets = extract_tweets(search_words,date_since,date_until,numTweets)
        # Store these tweets into a python list
        tweet_list = [tweet for tweet in tweets]
        print(len(tweet_list))
        noTweets = 0

        for tweet in tweet_list:
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            location = tweet.user.location
            following = tweet.user.friends_count
            followers = tweet.user.followers_count
            totaltweets = tweet.user.statuses_count
            usercreatedts = tweet.user.created_at
            tweetcreatedts = tweet.created_at
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']
            lst=[]
            for h in hashtags:
                lst.append(h['text'])
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text

            itweet = [username,acctdesc,location,following,followers,totaltweets,usercreatedts,tweetcreatedts,retweetcount,text,lst]
            db_tweets.loc[len(db_tweets)] = itweet

            noTweets += 1

            #filename = "tweets.csv"
            #with open(filename, "a", newline='') as fp:
             #   wr = csv.writer(fp, dialect='excel')
              #  wr.writerow(itweet)

        if i+1 != numRuns:
            time.sleep(920)

        filename = "static/analysis.csv"
        db_tweets['text'] = db_tweets['text'].apply(change)
        db_tweets = db_tweets[['retweetcount', 'text']]
        # Store dataframe in csv with creation date timestamp
        db_tweets.drop_duplicates(subset ="text", keep = 'first', inplace = True)
        db_tweets.to_csv(filename, index = False) #
        print('Scrapping Done')

# Functions for Sentiment Extractor
def save_weights(model, dst_fn):
    weights = model.get_weights()
    with open(dst_fn, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(model, weight_fn):
    with open(weight_fn, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

def loss_fn(y_true, y_pred):
    # adjust the targets for sequence bucketing
    ll = tf.shape(y_pred)[1]
    y_true = y_true[:, :ll]
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
        from_logits=False, label_smoothing=LABEL_SMOOTHING)
    loss = tf.reduce_mean(loss)
    return loss
 #Global Constants for TF sentiment extractor
MAX_LEN = 310
PAD_ID = 1
num_splits = 1
SEED = 88888

PATH = 'static/Tf-Roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)

def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 
    model.compile(loss=loss_fn, optimizer=optimizer)
    
    # this is required as `model.predict` needs a fixed size!
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    
    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])
    return model, padded_model

def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2",background_color='white', collocations=False).generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.tight_layout(pad=0)

@app.route("/",methods=['GET', 'POST'])
@app.route("/dashboard", methods=['GET', 'POST'])
def home():
    return render_template('dashboard.html')

@app.route("/live_case_count", methods=['GET', 'POST'])
def live_count():
    return render_template('live_case_count.html')

@app.route("/about_us", methods=['GET', 'POST'])
def about():
    return render_template('about_us.html')

HOUR = 3600;

thread = Thread()
thread_stop_event = Event()

model_type = 'roberta'
pretrained_model_name = 'roberta-base'

model_class, tokenizer_class, config_class = RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

pad_idx = transformer_tokenizer.pad_token_id

p = 'static/Roberta_Model'
learner = load_learner(p, 'transformer.pkl')

def plotGenerator():
    """
    Generates real time plots every 1 day.
    """
    while not thread_stop_event.isSet():
        # Initialise these variables:
        print('Code is Running!!!!!')

        search_words = "(#India AND #COVID-19) OR #COVID19India"
        yesterday = datetime.datetime.now() - datetime.timedelta(days = 1)
        date_since = yesterday.strftime("%Y-%m-%d")
        date_until = datetime.datetime.today().strftime('%Y-%m-%d')
        numTweets = 2000
        numRuns = 1
        # Call the function scraptweets
        program_start = time.time()
        scraptweets(search_words, date_since, date_until, numTweets, numRuns)
        program_end = time.time()

        path = 'static/analysis.csv'
        predictions = pd.read_csv(path)

        print('Start Prediction')

        predictions['Prediction'] = predictions['text'].apply(lambda x: predict_sentiment(learner, x))
        predictions['Prediction'] = predictions['Prediction'].apply(sentiment_label)
        class_names = ['negative','positive','neutral']

        predictions.rename(columns={'Prediction':'sentiment'},inplace=True)

        print('Predictions Done')

        path = 'static/analysis.csv'
        predictions.to_csv(path,index=False)
        test = pd.read_csv(path)

        MAX_LEN = 310
        PAD_ID = 1
        num_splits = 1
        SEED = 88888

        PATH = 'static/Tf-Roberta/'
        tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=PATH+'vocab-roberta-base.json', 
            merges_file=PATH+'merges-roberta-base.txt', 
            lowercase=True,
            add_prefix_space=True
        )

        test['len'] = test['text'].str.len()
        test = test[test['len']<=310]
        test.drop("len",axis=1,inplace=True)
        test.reset_index(drop=True, inplace=True)

        ct = test.shape[0]
        input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
        attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
        token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')
        sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

        for k in range(test.shape[0]):
                
            # INPUT_IDS
            text1 = " "+" ".join(test.loc[k,'text'].split())
            enc = tokenizer.encode(text1)                
            s_tok = sentiment_id[test.loc[k,'sentiment']]
            input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
            attention_mask_t[k,:len(enc.ids)+3] = 1

        DISPLAY=1 # USE display=1 FOR INTERACTIVE
        preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
        preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

        # for fold in range(0,5):
        K.clear_session()
        model, padded_model = build_model()
        path = 'static/R_CNN_weights/'
        weight_fn = path+'v0-roberta-0.h5'

        print('Loading model...')
        # model.load_weights('%s-roberta-%i.h5'%(VER,fold))
        load_weights(model, weight_fn) 

        print('Predicting Test...')

        preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
        preds_start += preds[0]/num_splits
        preds_end += preds[1]/num_splits

        all = []
        for k in range(input_ids_t.shape[0]):
            a = np.argmax(preds_start[k,])
            b = np.argmax(preds_end[k,])
            if a>b: 
                st = test.loc[k,'text']
            else:
                text1 = " "+" ".join(test.loc[k,'text'].split())
                enc = tokenizer.encode(text1)
                st = tokenizer.decode(enc.ids[a-2:b-1])
            all.append(st)
        
        test['selected_text'] = all
        test.to_csv('static/analysis.csv',index=False)

        print('Extraction Done')

        #Plots start
        data=pd.read_csv("static/analysis.csv")
        data['selected_text'] = data['selected_text'].astype(str)
        df = data.sentiment.value_counts()
        size = list(df.values)
        names = list(df.index)
        fig = plt.figure(figsize=(10,10))
        plt.xlabel("Sentiment",Fontsize = 16)
        plt.ylabel("Frequency",Fontsize = 16)
        sns.barplot(names,size,alpha = 0.8)
        fig.savefig("static/images/realtime/bar.png")
        
        df_new = pd.DataFrame(dict(
            r=list(df.values),
            theta=list(df.index)))
        plt.figure(figsize=(10,10))
        fig = px.line_polar(df_new, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        fig.write_image("static/images/realtime/radar_plot.png")

        # Pie chart
        labels = list(df.index)
        sizes = list(df.values)
        # only "explode" the 2nd slice 
        explode = (0.1, 0.1, 0.1)
        #add colors
        colors = ['#ff9999','#66b3ff','#99ff99']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.savefig('static/images/realtime/pie_chart.png')

        for i in range(3):
          Data= data[data["sentiment"]==df.index[i]]
          Word_frequency = pd.Series(' '.join(Data.selected_text).split()).value_counts()[:20]#Calculating the words frequency
          plt.figure(figsize=(25,10))
          plt.ylabel("Frequency",fontsize=16)
          plt.title("Sentiment Triggers")
          sns.barplot(Word_frequency.index,Word_frequency.values,alpha=0.8)
          plt.savefig("static/images/realtime/wordfrequency_"+df.index[i]+".png")

        for i in range(3):
            Analysis_Data = data
            data["selected_text"]=data["selected_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))
            Sentiment = Analysis_Data[Analysis_Data['sentiment'] == df.index[i]]#Creating the dataframe of having same sentiment
            Word_frequency = pd.Series(' '.join(Sentiment.selected_text).split()).value_counts()[:50]#Calculating the words frequency
            generate_wordcloud(Word_frequency.sort_values(ascending=False),data.index[i])
            plt.savefig("static/images/realtime/Wordcloud_" +df.index[i]+".png")

        data["text"]=data["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))
        bigrams = [b for l in data.text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        bigram_counts = collections.Counter(bigrams)
        bigram_df = pd.DataFrame(bigram_counts.most_common(10),
                                    columns=['bigram', 'frequency'])
        x =bigram_df.bigram
        y = bigram_df.frequency

        fig, ax = plt.subplots(1, 1, figsize = (20, 15), dpi=300)
        sns.barplot(x,y,alpha=0.8)
        plt.ylabel("Frequency",fontsize=16)
        ax.set_xlabel('')
        plt.savefig('static/images/realtime/bigram_freq.png')

        ext_data_negative = data[data["sentiment"]=='negative']
        ext_data_positive = data[data["sentiment"]=='positive']
        bigrams = [b for l in ext_data_positive.selected_text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        bigram_counts = collections.Counter(bigrams)
        bigram_df_positive = pd.DataFrame(bigram_counts.most_common(60),
                                    columns=['bigram', 'frequency'])
        bigrams = [b for l in ext_data_negative.selected_text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        bigram_counts = collections.Counter(bigrams)
        bigram_df_negative = pd.DataFrame(bigram_counts.most_common(80),
                                    columns=['bigram', 'frequency'])

        # Create network plot 
        G=nx.grid_2d_graph(2,2)

        pos = nx.fruchterman_reingold_layout(G,k=10,iterations=100)
        fig,ax = plt.subplots(figsize=(50,30)) 
        d = bigram_df_negative.set_index('bigram').T.to_dict('records')
        for k, v in d[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))
        pos = nx.fruchterman_reingold_layout(G,k=10,iterations=100) 
          
        nx.draw_networkx(G, pos,
                        font_size=16,
                        width=4,
                        edge_color='#e25a4b',
                        node_size=500,
                        title = "Negative Sentiment",
                        with_labels = False,
                        ax=ax)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)

        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,bbox=dict(facecolor='#ffcd94', alpha=0.4),
                    horizontalalignment='center', fontsize=35)
        plt.savefig("static/images/realtime/ext_negative.png")        
        fig,ax  = plt.subplots(figsize=(50,30))
        d = bigram_df_positive.set_index('bigram').T.to_dict('records')
        for k, v in d[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))
        pos = nx.fruchterman_reingold_layout(G,k=10,iterations=100) 
          
        nx.draw_networkx(G, pos,
                        font_size=16,
                        width=4,
                        edge_color='#999894',
                        node_size=500,
                        with_labels = False,
                        title = "Positve Sentiment",
                        ax=ax)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)

        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,bbox=dict(facecolor='#7c99d0', alpha=0.4),
                    horizontalalignment='center', fontsize=35)
        plt.savefig("static/images/realtime/ext_positive.png")

        data["text"]=data["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))
        bigrams = [b for l in data.text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        bigram_counts = collections.Counter(bigrams)
        bigram_df = pd.DataFrame(bigram_counts.most_common(60),
                                    columns=['bigram', 'frequency'])
        
        d = bigram_df.set_index('bigram').T.to_dict('records')
        # Create network plot 
        G = nx.Graph()
        for k, v in d[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))

        fig,ax = plt.subplots(figsize=(20,20))
        pos = nx.spring_layout(G,dim=2,k=5)

        # Plot networks
        nx.draw_networkx(G, pos,
                        font_size=12,
                        width=4,
                        edge_color='grey',
                        node_color='#4a4140',
                        node_size=500,
                        with_labels = False,
                        ax=ax)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)


        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,
                    bbox=dict(facecolor='#ffcd94', alpha=0.4),
                    horizontalalignment='center', fontsize=25)
            
        fig.savefig('static/images/realtime/network.png')

        fig = px.box(data, y="retweetcount",points="all")
        fig.update_layout(
            yaxis_title="Retweet Count",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        fig.write_image('static/images/realtime/retweet_count_boxplot.png')

        fig = px.box(data, y="sentiment",points="all")
        fig.update_layout(
            yaxis_title="Sentiment",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        fig.write_image('static/images/realtime/sentiment_boxplot.png')

        print('Plotting Done.')

        socketio.sleep(24*HOUR)

@app.route('/real_time_analysis', methods=['GET', 'POST'])
def realtime():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('real_time_analysis.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')
    #Start the plot generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = socketio.start_background_task(plotGenerator)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

twitter_thread = Thread()
twitter_thread_stop_event = Event()

def gettweets():

    while not twitter_thread_stop_event.isSet():
        print('Tweet feed starts')
        for tweet in tweepy.Cursor(t_api.search,q="#" + "COVID19India"  + " -filter:retweets",rpp=5,lang="en", tweet_mode='extended').items(20):
            temp = {}
            text = change(tweet.full_text)
            temp["text"] = text
            temp["username"] = tweet.user.screen_name
            text = change(tweet.full_text)
            prediction = predict_sentiment(learner,text)
            prediction = sentiment_label(prediction)
            temp["label"] = prediction
            socketio.emit('tweets', {'Text': temp['text'], 'Username': temp['username'], 'Sentiment': temp['label']}, namespace='/twitter')
        print('Twitter feed end')
        socketio.sleep(300)


@app.route("/twitter_live_feed", methods=['GET', 'POST'])
def twitter():
    return render_template('twitter_live_feed.html')

@socketio.on('connect', namespace='/twitter')
def twitter_connect():
    # need visibility of the global thread object
    global twitter_thread
    print(' Twitter Client connected')

    #Start the tweet feed thread only if the thread has not been started before.
    if not twitter_thread.isAlive():
        print("Starting Twitter Thread")
        thread = socketio.start_background_task(gettweets)

@socketio.on('disconnect', namespace='/twitter')
def twitter_disconnect():
    print('Twitter Client disconnected')

#Run APP 

if __name__ == '__main__':
    socketio.run(app)
