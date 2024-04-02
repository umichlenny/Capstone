# Stock price prediction based on the Transformer Structure

This repository contains a task utilizing the pretrained transformer architecture for stock price prediction. Our project intends to use the BERT model, commonly used in Natural Language Processing (NLP), for prediction purposes. BERT's internal Transformer architecture can effectively analyze time series tasks such as stock price prediction. 


# Downloading Dataset

Here, please make sure you have installed the yfinance and pandas libraries. Once you have selected the start date, end date, and ticker, running the above code will download the data for analysis.

```
import yfinance as yf
import pandas as pd
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers.Symbol.tolist()
start_date = '2005-01-01'
end_data = '2023-12-31'
ticker = 'AAPL'
df = yf.download(tickers, start = start_date, end = end_data)
df = df.stack().reset_index(level=1)
df = df[df['Ticker'] == ticker]
```





# Requirements and Installation
To run my code, you'll need the following libraries.

- [torch](https://pytorch.org/)<br>
- [transformers](https://huggingface.co/docs/transformers/index)<br>
- [pandas](https://pandas.pydata.org/)<br>
- [yifinance](https://pypi.org/project/yfinance/)<br>
- [numpy](https://numpy.org/)<br>
- [pandas](https://pandas.pydata.org/)<br>
- [matplotlib](https://matplotlib.org/)<br>
