# Sentiment Analysis with BERT

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Dataset Analysis](#dataset-analysis)
4. [Text to Tokens Conversion](#text-to-tokens-conversion)
5. [DataLoader and Train-Test Split](#dataloader-and-train-test-split)
6. [Tokenization of the Dataset](#tokenization-of-the-dataset)
7. [Model Building](#model-building)
8. [Fine-Tuning](#fine-tuning)
9. [Training and Evaluation](#training-and-evaluation)
10. [Model Evaluation](#model-evaluation)
11. [Prediction](#prediction)
12. [Using Pipeline for Prediction](#using-pipeline-for-prediction)
13. [Conclusion](#conclusion)

## Introduction

This project demonstrates how to perform sentiment analysis on a Twitter dataset using the BERT model. We will cover the entire process from data loading and preprocessing to model training, evaluation, and prediction.

## Setup

First, ensure you have the necessary libraries installed. You can install them using the following commands:

```bash
!pip install -U transformers
!pip install -U accelerate
!pip install -U datasets
!pip install -U bertviz
!pip install -U umap-learn
!pip install seaborn --upgrade
```

## Dataset Analysis

Load the dataset and perform initial analysis:

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_multi_class_sentiment.csv")

# Dataset shape
print(df.shape)

# Dataset information
print(df.info())

# Check for null values
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())

# Label distribution
print(df['label'].value_counts())
```

Visualize the frequency of classes:

```python
import matplotlib.pyplot as plt

label_counts = df['label_name'].value_counts(ascending=True)
label_counts.plot.barh()
plt.title("Frequency of classes")
plt.show()
```

Analyze the distribution of words per tweet:

```python
df['Words per Tweet'] = df['text'].str.split().apply(len)
df.boxplot("Words per Tweet", by="label_name")
```

## Text to Tokens Conversion

Convert text to tokens using the BERT tokenizer:

```python
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "I love machine learning! Tokenization is awesome!!"
encoded_text = tokenizer(text)
print(encoded_text)

print(len(tokenizer.vocab), tokenizer.vocab_size, tokenizer.model_max_length)
```

## DataLoader and Train-Test Split

Split the dataset into training, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label_name'])

print(train.shape, test.shape, validation.shape)
```

Convert the datasets to Hugging Face `Dataset` objects:

```python
from datasets import Dataset, DatasetDict

dataset = DatasetDict(
    {'train': Dataset.from_pandas(train, preserve_index=False),
     'test': Dataset.from_pandas(test, preserve_index=False),
     'validation': Dataset.from_pandas(validation, preserve_index=False)
    }
)
print(dataset)
```

## Tokenization of the Dataset

Tokenize the dataset:

```python
def tokenize(batch):
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp

print(tokenize(dataset['train'][:2]))
emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)
print(emotion_encoded)
```

Create label mappings:

```python
label2id = {x['label_name']: x['label'] for x in dataset['train']}
id2label = {v: k for k, v in label2id.items()}

print(label2id, id2label)
```

## Model Building

Load the BERT model:

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(model_ckpt)
print(model.config)
```

## Fine-Tuning

Fine-tune the BERT model for sequence classification:

```python
from transformers import AutoModelForSequenceClassification, AutoConfig

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
print(model.config)
```

## Training and Evaluation

Set up training arguments:

```python
from transformers import TrainingArguments

batch_size = 64
training_dir = "bert_base_train_dir"
training_args = TrainingArguments(
    output_dir=training_dir,
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    disable_tqdm=False
)
```

Define compute metrics functions:

```python
!pip install evaluate
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics_evaluate(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

Build the trainer:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotion_encoded['train'],
    eval_dataset=emotion_encoded['validation'],
    tokenizer=tokenizer
)

trainer.train()
```

## Model Evaluation

Evaluate the model on the test set:

```python
preds_output = trainer.predict(emotion_encoded['test'])
print(preds_output.metrics)

y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = emotion_encoded['test'][:]['label']

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
print(label2id)
```

Plot the confusion matrix:

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, xticklabels=label2id.keys(), yticklabels=label2id.keys(), fmt='d', cbar=False, cmap='Reds')
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
```

## Prediction

Make predictions using the trained model:

```python
text = "I am super happy today I got it done finally"

def get_prediction(text):
    input_encoded = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]

print(get_prediction(text))
```

## Using Pipeline for Prediction

Use the Hugging Face pipeline for predictions:

```python
from transformers import pipeline

classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
print(classifier([text, 'hello, how are you?', "love you", "i am feeling low"]))
```

## Conclusion

This project demonstrates the complete process of performing sentiment analysis using the BERT model. We covered data loading, preprocessing, model training, evaluation, and prediction. The model achieved good performance on the test set, and we visualized the results using a confusion matrix.