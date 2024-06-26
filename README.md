# Stock price prediction based on the Transformer Structure

This repository contains a task utilizing the pretrained transformer architecture for stock price prediction. Our project intends to use the BERT model, commonly used in Natural Language Processing (NLP), for prediction purposes. BERT's internal Transformer architecture can effectively analyze time series tasks such as stock price prediction. 

## Requirements and Installation
To run my code, you'll need the following libraries.

- [torch](https://pytorch.org/)<br>
- [transformers](https://huggingface.co/docs/transformers/index)<br>
- [pandas](https://pandas.pydata.org/)<br>
- [yifinance](https://pypi.org/project/yfinance/)<br>
- [numpy](https://numpy.org/)<br>
- [pandas](https://pandas.pydata.org/)<br>
- [matplotlib](https://matplotlib.org/)<br>

## Get Started

### Import requried libraries
```python
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
### Downloading Dataset
Here, please make sure you have installed the yfinance and pandas libraries and then we can download data from different time points for analysis by adjusting the following settings."

- start date
- end date
- ticker

```python
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers.Symbol.tolist()
start_date = '2005-01-01'
end_data = '2023-12-31'
ticker = 'AAPL'
df = yf.download(tickers, start = start_date, end = end_data)
df = df.stack().reset_index(level=1)
df = df[df['Ticker'] == ticker]
```

### Choose pre-trained Mode

We choose best-base-uncased as our base. However, we can adjust some underlying architectures by adjusting settings, such as

- num_layers
- hidden_size
- dropout

```python
class StockPricePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers = 5, hidden_size = 256, dropout = 0.7):
        super(StockPricePredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
```

### Feature selection

The following input features are selected for analysis.
- Ticker
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
        text = f"{item['Ticker']} {item['Adj Close']} {item['Volume']} {item['Close']} {item['High']} {item['Low']} {item['Open']}"
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['Adj Close'], dtype=torch.float32)
        }
```

### Adjust training settings
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
NUM_EPOCHS = 7
```

### Calculate the loss
Calculate AVERAGE_LOSS; if AVERAGE_LOSS is lower than BEST_LOSS, save the best Model run of this epoch
```python
for EPOCH in range(NUM_EPOCHS):
    TOTAL_LOSS = 0.0
    print(f"Epoch {EPOCH+1}/{NUM_EPOCHS}")
    for BATCH_INDEX, BATCH in enumerate(DATALOADER):
        input_ids = BATCH['input_ids']
        attention_mask = BATCH['attention_mask']
        labels = BATCH['labels']
        OPTIMIZER.zero_grad()
        outputs = MODEL(input_ids = input_ids, attention_mask = attention_mask)
        LOSS = CRITERION(outputs.squeeze(1), labels)
        LOSS.backward()
        OPTIMIZER.step()
        TOTAL_LOSS += LOSS.item()
    AVERAGE_LOSS = TOTAL_LOSS / len(DATALOADER)

    if AVERAGE_LOSS < BEST_LOSS:
        BEST_LOSS = AVERAGE_LOSS
        BEST_MODEL_STATE_DICT = MODEL.state_dict()

    TRAIN_LOSS.append(AVERAGE_LOSS)
    print(f"Average Loss: {AVERAGE_LOSS:.4f}")
```
### Plot the loss during the training process
```python
plt.plot(TRAIN_LOSS, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
```
![CleanShot 2024-04-06 at 13 32 03@2x](https://github.com/umichlenny/Capstone/assets/149079836/bbf0827b-9012-41b6-afb4-e731f05e2f37)
### Model Evaluation 
Using the optimal model to compare estimated prices with actual prices.
```python
if BEST_MODEL_STATE_DICT:
    MODEL.load_state_dict(BEST_MODEL_STATE_DICT)

MODEL.eval()
TEST_LOSSES = []

PREDICTED_PRICES = []
ACTUAL_PRICES = []

DATE_LABELS = TESTING_DATA.index.tolist()
DATE_LABELS = DATE_LABELS[:50]

TEST_DATASET = StockPriceDataset(TESTING_DATA.head(50), TOKENIZER)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=10, shuffle=False)

with torch.no_grad():
    for BATCH_INDEX, BATCH in enumerate(TEST_DATALOADER):
        INPUT_IDS = BATCH['input_ids']
        ATTENTION_MASK = BATCH['attention_mask']
        LABELS = BATCH['labels']
        OUTPUTS = MODEL(input_ids = INPUT_IDS , attention_mask = ATTENTION_MASK)
        PREDICTED_PRICES.extend(OUTPUTS.squeeze(1).tolist())
        ACTUAL_PRICES.extend(LABELS.tolist())
        LOSS = CRITERION(OUTPUTS.squeeze(1), LABELS)
        TEST_LOSSES.append(LOSS.item())

plt.figure(figsize=(10, 5))
plt.plot(DATE_LABELS, PREDICTED_PRICES, label='Predicted Prices', color='green')
plt.plot(DATE_LABELS, ACTUAL_PRICES, label='Actual Prices', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Price vs Actual Price')
plt.legend()
plt.show()
```
![CleanShot 2024-04-06 at 13 30 35@2x](https://github.com/umichlenny/Capstone/assets/149079836/62e60507-6289-4b80-9718-ad2f98dede38)



### Model Evaluation
Assessing whether the model has been trained successfully using residual plots, and whether data points are randomly distributed around y=0.
```python
plt.figure(figsize = (10, 6))
plt.scatter(ACTUAL_PRICES, np.array(ACTUAL_PRICES) - np.array(PREDICTED_PRICES), color='green', alpha=0.7)
plt.axhline(y=0, color= 'blue', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('The comparision of Residuals and Actual Values')
plt.grid(True)
plt.show()
```
![CleanShot 2024-04-06 at 13 28 12@2x](https://github.com/umichlenny/Capstone/assets/149079836/099bbfa9-fcfe-4541-aff5-a06209a37b12)


### Model Evaluation
R-square indicates the extent to which the model explains the variance of the target variable. A value closer to 1 signifies a better model explanation
```python
from sklearn.metrics import r2_score
import torch
TEST_DATASET = StockPriceDataset(TESTING_DATA, TOKENIZER)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=32, shuffle=False)
MODEL.eval()
PREDICTIONS = []
ACTUALS = []
with torch.no_grad():
    for BATCH in TEST_DATALOADER:
        INPUT_IDS = BATCH['input_ids']
        ATTENTION_MASK = BATCH['attention_mask']
        LABELS = BATCH['labels'].numpy()
        OUTPUT = MODEL(input_ids = INPUT_IDS, attention_mask = ATTENTION_MASK)
        PREDICTION = OUTPUT.detach().numpy().flatten()
        PREDICTIONS.extend(PREDICTION)
        ACTUALS.extend(LABELS)
r2 = r2_score(ACTUALS, PREDICTIONS)
print(f"R-squared: {r2}")
```
R-squared: 0.41825668042162156

