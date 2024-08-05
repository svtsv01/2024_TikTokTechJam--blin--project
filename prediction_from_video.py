import pyktok as pyk
from pydub import AudioSegment
import os
from moviepy.editor import *
from vosk import Model, KaldiRecognizer
import wave
import json
import fnmatch
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import warnings
warnings.filterwarnings("ignore")

path_to_dir = 'C:/Users/pogiz/OneDrive/PC/Projects/2024_TikTokTechJam--blin--project/Prediction'
tt_link = 'https://www.tiktok.com/@textplot/video/7399394583483813153?_r=1&_t=8ocAbEBuxya'

#Load video from TikTok
pyk.specify_browser('chrome')
pyk.save_tiktok(tt_link, True)

for file in os.listdir('.'):
    if fnmatch.fnmatch(file, '*@*'):
        video_path = file
        break

#Extract MP3 from video
video = VideoFileClip(video_path)
audio = video.audio
audio.write_audiofile("output_audio.mp3")


# Load MP3 file, convert to wav and set parameters
audio = AudioSegment.from_mp3("output_audio.mp3")
audio.export("output_audio.wav", format="wav")
audio = AudioSegment.from_wav("output_audio.wav")
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)
audio.export("output_audio_mono.wav", format="wav")


model = Model('vosk-model-en-us-0.22')
rec = KaldiRecognizer(model, 16000)

# Open WAV file
wf = wave.open("output_audio_mono.wav", "rb")

# List to hold all text segments
transcribed_text_list = []

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        transcribed_text_list.append(result['text'])

# Handle last part of audio
final_result = json.loads(rec.FinalResult())
transcribed_text_list.append(final_result['text'])

# Concatenate all text segments
complete_text = ' '.join(transcribed_text_list)

print(complete_text)

# Load data
test_text = [complete_text]

# Load BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device("cpu")

# Tokenize and encode sequences
max_length = 128
tokens_test = tokenizer.batch_encode_plus(test_text, max_length=max_length, pad_to_max_length=True, truncation=True)

# Convert lists to tensors
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

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

# Load best model and evaluate data
model.load_state_dict(torch.load('saved_weights.pt', map_location=torch.device('cpu')))
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(preds[0])

#Remove the files
video.close()
wf.close()
os.remove('output_audio.mp3')
os.remove('output_audio.wav')
os.remove('output_audio_mono.wav')
os.remove(video_path)