import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from transformers import pipeline
n = 50
tickers = ['AAPL']

finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finwiz_url + ticker + "&p=d"
    req = Request(url=url, headers={'user-agent': '	31012b3854f54e58ba82f715cdd449a4'})
    resp = urlopen(req)    
    html = BeautifulSoup(resp, features="html.parser")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

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

   
analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)

scores = []   
for i, headline in enumerate(news['Headline']):
    print(f"\nAnalysing {i+1}/{news['Headline'].count()} {round((i+1)/news['Headline'].count(),3)*100}%")
    score = analyzer(headline)#{"label":""}]
    if score[0]["label"] == "positive":
        scores.append(score[0]["score"])
        print(f"{headline}")
        print(score[0]["score"])
    elif score[0]["label"] == "negative":
        scores.append(-score[0]["score"])
        print(f"{headline}")
        print(-score[0]["score"])
    elif score[0]["label"] == "neutral":
        scores.append(0)
        print(f"{headline}")
        print(0)
    if i == n*len(tickers):
        break
        
df_scores = pd.DataFrame(scores)

news = news.join(df_scores, rsuffix='_right')
# View Data 
news['Date'] = pd.to_datetime(news.Date).dt.date
unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}
values = []

for ticker in tickers: 
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.rename(columns={0:"Score"})
    print (dataframe.head())
    dataframe = dataframe.drop(dataframe[dataframe["Score"] == 0].index)
    print (dataframe.head())
    mean = round(dataframe["Score"].mean(), 2)
    values.append(mean)
    
df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment']) 
df = df.set_index('Ticker')
df = df.sort_values('Mean Sentiment', ascending=False)
print (df)