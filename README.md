# SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19
## Twitter Sentiment Analysis &amp; Extraction for COVID-19
<p align="center">
  <img src="https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Pictures/Twitterlogo.png"/>
</p>
   
Hello there! This is the Github repository for **CoVis** - The Visualisation Dashboard of COVID-19 Twitter Sentiment Analysis. This project is based on analyzing the sentiment of tweets which help in understanding the pulse of the nation towards the pandemic.
Tweet is a post on the social media platform Twitter with a maximum of 140 characters. In this project the data for training was obtained from IEEE Datasets and tweets scrapped from Twitter over a period of 3 months using the Tweepy API. Transfer learning approach was adopted in building the models in the project.
Two deep learning models were used,namely:
1) RoBERTa - For Sentiment Analysis
2) RoBERTa-CNN - For Sentiment Triggers Extraction.

## Website UI 

 ## Public Sentiment Analysis   
 To create a public sentiment analysis dashboard, tweets were scraped on the days when the Government of India made major decisions like the “lockdown”, “lockdown 2.0”,  “unlock1.0”, etc based on the hashtags used in the tweets, during a period of 3 months using the Tweepy API and GetOldTweets3 package. A new data set was created using these  tweets to create a public sentiment dashboard on the final web application which depicted categories like sentiment triggers, overall sentiment of the tweets using robust graphs like the funnel chart, word nexus plots, bigram frequency, box plots and Network Visualizations etc.
 ![Public Sentimental Analysis](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Pictures/public.gif)

 ## Real Time Sentiment Analysis
Further creating a real-time sentiment analysis dashboard, COVID-19 tweets in India, the Tweepy API is used to scrape tweets from twitter on a real-time basis. 2,500 tweets are scraped everyday using this API, and their sentiment is then extracted using the developed language model. The analysed results are then displayed on the website using various graphs as in the case of public sentiment analysis dashboard. This dashboard updates itself every 24 hours.

 ## Twitter Live Feed Analysis
In order to create the Twitter Live Feed Analysis Dashboard the website scrapes tweets in real-time every 5 minutes, performs sentiment analysis of these tweets and displays a 3-D interactive, live, donut plot depicting the sentiment counts of the tweets. The analysed tweets are displayed on the page with the tweet text and the tweet sentiment. Also the real time tweets are displayed using the publish functionality of twitter.

 ## Live Case Count 
This section of the dashboard is for displaying the live, state wise statistics of COVID-19 cases in India.


## Development Phase

For the Model Training and Validation the fastai approach was used along with Keras(Tensorflow 2.0) and Pytorch. The development phase of the project is divided in 5 phases:
1) [Data Collection and Cleaning.](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Notebooks/Data_Creation_%26_Cleaning.ipynb)
2) [Exploratory Data Analysis and Preprocessing.](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Notebooks/Exploratory_Data_Analysis.ipynb)
3) [Roberta Model Training](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Notebooks/Roberta_Model_Training.ipynb),[Roberta CNN Sentiment Extractor](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Notebooks/Roberta_CNN.ipynb) & [Sentiment Analyzer.](https://github.com/SmartPracticeschool/SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19/blob/master/Notebooks/Sentiment_Analyzer.ipynb)
4) Development of Web App based on Flask Socket.IO.
5) Deployment of WebApp on GCP(Google Cloud Platform)

## Repository Structure & Files
The structure of the repository is represented below:

    |
    |
    |
    |
## Dockerfile & Requirements.txt

The github repo is provided with a dockerfile and requirements.txt file to recreate the app deployed in the project. 
The dockerfile creates a virtual environment with required python version and packages for web app deployment. The required Python version is 3.6.X. All the dependencies required for the code in the repo can be installed using requirements.txt.

    pip install -q -r requirements.txt
    
The Development of the website is divided into 4 phases:

1) Public Sentiment Analysis
2) Real Time Sentiment Analysis
3) Twitter Live Feed Analysis
4) Live Case count    
    
## Notebooks

## Data Creation and Data Cleaning

Initially, the IEEE Coronavirus (COVID-19) Tweets Data set was downloaded from their website. Upon inspection, it was found that many tweets did not have geo-location tags and also many were in different languages apart from English. Due to this challenge in obtaining proper data, a new data set named Geo-Tagged Coronavirus (COVID-19) Tweets Data set was obtained from the same website. These tweets were then hydrated using the “Hydrator” software and also a few python commands. Then, tweets in “English” and tweets from “India” were randomly chosen and a new dataset was created. Further, another data set containing COVID-19 related tweets from India were obtained from Kaggle.This data set was then cleaned and normalised to make it useful for further analysis.


## Exploratory Data Analysis
 
The data set thus obtained after cleaning  was then subjected to Exploratory Data Analysis (EDA) by plotting various types of graphs based on the sentiments and sentiment triggers, to gain valuable insights from the data. The frequency distribution graphs gives us a good perspective of the dataset and also gives us an insight into predicting the model's generalization capability. By plotting the graphs on the basis of sentiments and the sentiment triggers, it was clear that there was not much of a difference in the sentiment trend in tweets prevalent in India when compared to the rest of the world. From further analysis of the data it was evident that people mostly expressed neutral or positive about the pandemic, with only a very few people being negative towards it. This could be attributed to the fact that initially there was much chaos and panic around the pandemic, but over time, people have become less skeptic and more accustomed to the pandemic, which could have led to the trends being more deviated towards the positive and neutral end. The initial assumption about terms like positive and negative conveying the opposite meaning was opposed to the observed trends after the analysis and hence the assumption was dropped. Statistical analysis of the data set was also carried out to investigate further into the data. Feature selection and feature engineering were carried out on the data set for further analysis.

## Roberta Model Training & Sentiment Analyzer
Transfer learning methods were implemented to carry out sentiment analysis. Sentiment Analysis of Tweets was carried out by integrating and using both the Huggingface Transformer Library and FastAI. Further Slanted Triangular Learning Rates, Discriminate Learning Rate and even Gradual Unfreezing were used, as a result of which, state-of-the-art results were obtained rapidly without even tuning the parameters. The Data obtained from the previous process was then tokenised and passed through the model for Sentiment analysis. This yielded a model with an accuracy of **97%** over the data set.The Tweepy API was used to scrape tweets in real time which were then passed through the model to obtain the sentiments.

## Roberta-CNN Sentiment Extractor
After the completion of the sentiment analysis the data was further explored for the sentiment triggers in the tweets. HuggingFace transformers don't have a TFRobertaForQuestionAnswering, for this purpose, a TFRobertaModel was created to convert trained data into arrays that the Roberta model can interpret.While training the Sentiment Extractor model, 5 stratified KFolds were used in such a way that, in each fold the best model weights were saved and these weights were reloaded before carrying out testing and predictions. Roberta with CNN head was used for Twitter Sentiment Extraction. Thus after passing the data through this model we obtained a new column of the extracted text for the sentiments which was also used to plot certain graphs.

## Flask App
A flask app was used for setting up website routing. It is used to integrate the back end machine learning models with the dashboard. Then Socketio (web sockets) were used for dynamic implementations on the website, namely the Real-Time Plot Generators and Twitter live feed. The basic functionality of the Flask Socketio lies in running background threads when the client is not connected to the website thereby enabling dynamic plotting.





