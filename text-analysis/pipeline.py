import whisper
from transformers import pipeline
unmasker = pipeline('fill-mask', model='xlm-roberta-base')
print(unmasker("Hello I'm a <mask> model."))

# w_model = whisper.load_model("base")
# audio_path = "/Users/hongtan/Desktop/DSClub_Project/speech-to-text/audio-test-files/insane.wav"
# result = w_model.transcribe(audio_path, fp16=False)
# text = result["text"]
# print(text)

# classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')
# prediction = classifier(text)

# print(prediction)