# Stock price prediction based on the Transformer Structure

This repository contains a task utilizing the pretrained transformer architecture for stock price prediction. Our project intends to use the BERT model, commonly used in Natural Language Processing (NLP), for prediction purposes. BERT's internal Transformer architecture can effectively analyze time series tasks such as stock price prediction. 


# Downloading Dataset

Here, please make sure you have installed the yfinance and pandas libraries. Once you have selected the start date, end date, and ticker, running the above code will download the data for analysis.

```python
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

# Get Started

## Choose pre-trained Mode

We choose best-base-uncased as our base. However, we can adjust some underlying architectures by adjusting settings, such as

- num_layers
- hidden_size
- dropout

```python
class StockPricePredictionModel(nn.Module):
    #num_layers =3^
    #hidden_size=128^
    #dropout=0.5^
    def __init__(self, input_size, output_size, num_layers = 5, hidden_size = 256, dropout = 0.7):
        super(StockPricePredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
```
