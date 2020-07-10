# SBSPS-Challenge-2700-Twitter-Sentiment-Analysis-Extraction-for-COVID-19
## Twitter Sentiment Analysis &amp; Extraction for COVID-19

Hello there! This is the Github repository for **CoVis** - The Visualisation Dashboard of COVID-19 Twitter Sentiment Analysis. This project is based on analyzing the sentiment of tweets which help in understanding the pulse of the nation towards the pandemic.
Tweet is a post on the social media platform Twitter with a maximum of 140 characters. In this project the data for training was obtained from IEEE Datasets and tweets scrapped from Twitter over a period of 3 months using the Tweepy API. Transfer learning approach was adopted in building the models in the project.
Two deep learning models were used,namely:
1) RoBERTa - For Sentiment Analysis
2) RoBERTa-CNN - For Sentiment Triggers Extraction.

For the Model Training and Validation the fastai approach was used along with Keras(Tensorflow 2.0) and Pytorch. The development phase of the project is divided in 5 phases:
1) Data Collection and Cleaning
2) Exploratory Data Analysis and Preprocessing.
3) Sentiment Analysis and Sentiment Extractor models training and validation.
4) Development of Web App based on Flask Rest API
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
    
## Notebooks



## Exploratory Data Analysis
