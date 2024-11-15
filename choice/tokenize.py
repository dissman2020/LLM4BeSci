#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
torch.manual_seed(42) # For reproducibility

# # Processing data
# Reading in the .csv data
df = pd.read_csv('choice.csv', index_col=0)

# Add column with prompts
num_participants = df.participant.max() + 1
num_tasks = df.task.max() + 1
instructions = "You made the following observations in the past:\n"
question = "Q: Which machine do you choose?\nA: Machine"
text = []
for participant in tqdm(range(num_participants)):
    df_participant = df[(df['participant'] == participant)]

    for task in range(num_tasks):
        # new prompt for each task
        history = ""
        df_task = df_participant[(df_participant['task'] == task)]
        num_trials = df_task.trial.max() + 1
        for trial in range(num_trials):
            df_trial = df_task[(df_task['trial'] == trial)]
            # add text for free choice trials
            if not df_trial['forced_choice'].item():
                trials_left = num_trials - trial
                trials_left = str(trials_left) + " additional choices" if trials_left > 1 else str(trials_left) + " additional choice"
                trials_left_string = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n\n"

                prompt = instructions + history + "\n" + trials_left_string + question
                text.append(prompt)
            else:
                text.append("")

            # add data to history
            c = df_trial.choice.item()
            r = df_trial.reward.item()
            history += "- Machine " + str(c+1) +  " delivered " + str(r) + " dollars.\n"

df['text'] = text

# Removing forced choice trials and converting into a HuggingFace dataset
df = df[~df.forced_choice]
dat = Dataset.from_pandas(df)

model_ckpt = '/data/kankan.lan/llms/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(f'Vocabulary size: {tokenizer.vocab_size}, max context length: {tokenizer.model_max_length}')

# Function to tokenize a batch of samples
batch_tokenizer = lambda batch: tokenizer(batch['text'], padding=True, truncation=True)
#  Tokenizing the dataset
dat = dat.map(batch_tokenizer, batched=True, batch_size=None)
# Setting the format of the dataset to torch tensors for passing to the model
dat.set_format('torch', columns=['input_ids', 'attention_mask'])

# # Loading the model for feature extraction
# Loading the model and moving it to the GPU if available
if torch.cuda.is_available():  # for nvidia GPUs
    device = torch.device('cuda')
elif torch.backends.mps.is_available(): # for Apple Metal Performance Sharder (mps) GPUs
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# Loading distilbert-base-uncased and moving it to the GPU if available
model = AutoModel.from_pretrained(model_ckpt).to(device)
print(f'Model inputs: {tokenizer.model_input_names}')

def extract_features(batch):
    """Extract features from a batch of items"""
    inputs = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

dat = dat.map(extract_features, batched=True, batch_size=8)
print(dat['hidden_state'].shape)

features = pd.DataFrame(dat['hidden_state'])
print(features)

# 保存两个 Tensor 为 .pth 文件
torch.save(dat['choice'],"choice.pth")
torch.save(features,"features.pth")
