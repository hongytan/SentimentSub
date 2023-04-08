from transformers import pipeline

colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#ff0000', 'sadness':'#00ff00'}
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')
# prediction = classifier("Hello World")

# Read and classify each line of dialogue
with open('/Users/hongtan/Desktop/sentimentsub/audio.srt', 'r') as f:
    lines = f.readlines()
    n = len(lines)
    for i in range(2,n,4):
        label = classifier(lines[i])[0]['label']
        color = colors[label]
        new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
        lines[i] = new_line

with open('/Users/hongtan/Desktop/sentimentsub/audio.srt', 'w') as f:
    f.writelines(lines)


# print(prediction)