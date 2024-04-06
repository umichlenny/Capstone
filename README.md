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

### Downloading Dataset

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

### Choose pre-trained Mode

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

### Feature selection

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
NUM_EPOCHS = 10
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
![CleanShot 2024-04-05 at 22 23 28@2x](https://github.com/umichlenny/Capstone/assets/149079836/d3b22ccd-56fe-4671-a246-5714c6d18c06)




```python
plt.plot(TRAIN_LOSS, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
```
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
DATE_LABELS = DATE_LABELS[:40]

TEST_DATASET = StockPriceDataset(TESTING_DATA.head(40), TOKENIZER)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=10, shuffle=False)

with torch.no_grad():
    # BATCH_INDEX^
    # BATCH^
    for BATCH_INDEX, BATCH in enumerate(TEST_DATALOADER):
        # INPUT_IDS^
        # ATTENTION_MASK^
        INPUT_IDS = BATCH['input_ids']
        ATTENTION_MASK = BATCH['attention_mask']
        # LABELS^
        LABELS = BATCH['labels']
        # OUTPUTS^
        OUTPUTS = MODEL(input_ids = INPUT_IDS , attention_mask = ATTENTION_MASK)
        PREDICTED_PRICES.extend(OUTPUTS.squeeze(1).tolist())
        ACTUAL_PRICES.extend(LABELS.tolist())
        # LOSS^
        LOSS = CRITERION(OUTPUTS.squeeze(1), LABELS)
        TEST_LOSSES.append(LOSS.item())

plt.figure(figsize=(10, 5))
plt.plot(DATE_LABELS, PREDICTED_PRICES, label='Predicted Prices', color='blue')
plt.plot(DATE_LABELS, ACTUAL_PRICES, label='Actual Prices', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted vs Actual Prices')
plt.legend()
plt.show()
```
![CleanShot 2024-04-05 at 22 23 38@2x](https://github.com/umichlenny/Capstone/assets/149079836/e51ddee0-7e2e-430d-9061-e108b6110646)


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



