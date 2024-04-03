# Stock price prediction based on the Transformer Structure

This repository contains a task utilizing the pretrained transformer architecture for stock price prediction. Our project intends to use the BERT model, commonly used in Natural Language Processing (NLP), for prediction purposes. BERT's internal Transformer architecture can effectively analyze time series tasks such as stock price prediction. 

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

# Downloading Dataset

Here, please make sure you have installed the yfinance and pandas libraries and then wcan download data from different time points for analysis by adjusting the following settings."

- start date
- end date
- ticker

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

## Choose pre-trained Mode

We choose best-base-uncased as our base. However, we can adjust some underlying architectures by adjusting settings, such as

- num_layers
- hidden_size
- dropout

```python
class StockPricePredictionModel(nn.Module):
    #num_layers =3^
    def __init__(self, input_size, output_size, num_layers = 5, hidden_size = 256, dropout = 0.7):
        super(StockPricePredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
```

## Feature selection

The following input features are selected for analysis.
- Adj Close
- Volume
- Close
- High
- Low
- Open

```python
class StockPriceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = f"{item['Adj Close']} {item['Volume']} {item['Close']} {item['High']} {item['Low']} {item['Open']}"
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['Adj Close'], dtype=torch.float32)
        }
```

## Adjust training settings
We can adjust the following settings to achieve different training processes.
- TRAINING_RATIO
- NUM_EPOCHS
```python
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = StockPricePredictionModel(input_size=768, output_size=1)
SORTED_DATAA = data.sort_values(by='Date')
TRAINING_RATIO = 0.6
TRAINING_SIZE = int(TRAINING_RATIO * len(SORTED_DATAA))
TRAINING_INDICES = np.arange(TRAINING_SIZE)
TESTING_INDICES = np.arange(TRAINING_SIZE, len(SORTED_DATAA))
TRAINING_DATA = SORTED_DATAA.iloc[TRAINING_INDICES]
TESTING_DATA = SORTED_DATAA.iloc[TESTING_INDICES]
DATASSET = StockPriceDataset(TRAINING_DATA, TOKENIZER)
DATALOADER = DataLoader(DATASSET, batch_size=16, shuffle=True)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-4)
CRITERION = nn.MSELoss()
TRAIN_LOSS = []
BEST_MODEL_STATE_DICT = None
BEST_LOSS = float('inf')
NUM_EPOCHS = 10
```
