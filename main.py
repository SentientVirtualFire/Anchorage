import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from transformers import pipeline
import requests, os

API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
headers = {"Authorization": f"Bearer {os.environ['huggingface']}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I like you. I love you",
})
# Parameters 
n = 3 #the # of article headlines displayed per ticker
tickers = ['AAPL']

# Get Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finwiz_url + ticker + "&p=d"
    req = Request(url=url, headers={'user-agent': '	31012b3854f54e58ba82f715cdd449a4'})
    resp = urlopen(req)    
    html = BeautifulSoup(resp, features="html.parser")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text() 
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]
        
        parsed_news.append([ticker, date, time, text])
        
# Sentiment Analysis
analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
scores1 = news['Headline'].apply(analyzer).tolist()
scores2 = []
for score in scores1:
    if score[0]["label"] == "positive":
        scores2.append(score[0]["score"])
    if score[0]["label"] == "negative":
        scores2.append(-score[0]["score"])
    if score[0]["label"] == "neutral":
        scores2.append(0)

df_scores = pd.DataFrame(scores2)
news = news.join(df_scores, rsuffix='_right')

# View Data 
news['Date'] = pd.to_datetime(news.Date).dt.date

unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers: 
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    print (dataframe)
    dataframe = dataframe.drop(columns = ['Headline'])
    print ('\n')
    print (dataframe.head(30))
    
    mean = round(dataframe[0].mean(), 2)
    values.append(mean)
    
df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment']) 
df = df.set_index('Ticker')
df = df.sort_values('Mean Sentiment', ascending=False)
print ('\n')
print (df)