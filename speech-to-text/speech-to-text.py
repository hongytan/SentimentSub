import stable_whisper
import whisper
import re
from transformers import pipeline

model = whisper.load_model("tiny")
model = stable_whisper.load_model('base')

mp4_file = "./audio-test-files/Friends Joey's Bad Birthday Gift.mp4"
result = model.transcribe(mp4_file, fp16=False)

result2 = model.transcribe(mp4_file, fp16=False)
result2.save_as_json('audio.json')

text = result['text']
# result.save_as_json('audio.json')

res = re.split('[?.,!]', text)

# for text in res[:5]:
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')
predictions = classifier(res)

labels = []
for prediction in predictions:
    labels.append(prediction['label'])
print(labels)