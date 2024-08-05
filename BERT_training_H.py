import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import pandas as pd

print('lessgo')

# Load data
train_text = pd.read_csv('train_text.tsv', sep='\t', names=['text'])['text']
train_labels = pd.read_csv('train_labels.tsv', sep='\t', names=['label'])['label']
val_text = pd.read_csv('val_text.tsv', sep='\t', names=['text'])['text']
val_labels = pd.read_csv('val_labels.tsv', sep='\t', names=['label'])['label']
#test_text = pd.read_csv('test_text.tsv', sep='\t', names=['text'])['text']
#test_labels = pd.read_csv('test_labels.tsv', sep='\t', names=['label'])['label']

# Load BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device("cuda")

# Tokenize and encode sequences
print('tokenizing...')
max_length = 128
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
print('train done')
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
print('val done')
#tokens_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
print('finished tokenizing')

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())
# test_seq = torch.tensor(tokens_test['input_ids'])
# test_mask = torch.tensor(tokens_test['attention_mask'])
# test_y = torch.tensor(test_labels.tolist())

# Create dataloaders
batch_size = 16
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Model definition
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = BERT_Arch(bert)
model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * 10  # Adjust number of epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Compute class weights and define loss function
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
cross_entropy = nn.NLLLoss(weight=weights)

# Training function
def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Evaluation function
def evaluate():
    print("\nEvaluating...")
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Training and evaluation
start = time.time()
best_valid_loss = float('inf')
epochs = 3
train_losses, valid_losses = [], []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    valid_loss, _ = evaluate()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

stop = time.time()
print(f'This took {stop-start} seconds')

#Load best model and evaluate on test data
# model.load_state_dict(torch.load('saved_weights.pt'))
# with torch.no_grad():
#     preds = model(test_seq.to(device), test_mask.to(device))
#     preds = preds.detach().cpu().numpy()

# preds = np.argmax(preds, axis=1)
# print(classification_report(test_y, preds))