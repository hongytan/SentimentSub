from transformers import pipeline
import os

def Upload():

    colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#ff0000', 'sadness':'#00ff00'}
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')
    # prediction = classifier("Hello World")

    #in sentimentsub directory
    output_file = './audio.srt'
    srt_file = './audio.srt'
    mp4_file = './website/sentsub/media/videos/Friends_Joeys_Bad_Birthday_Gift.mp4'

    os.system(f'stable-ts {mp4_file} -o {output_file} --word_level False --fp16 False -y')
    
    # Read and classify each line of dialogue
    with open(srt_file, 'r') as f:
        lines = f.readlines()
        n = len(lines)
        for i in range(2,n,4):
            label = classifier(lines[i])[0]['label']
            color = colors[label]
            new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
            lines[i] = new_line

    with open(srt_file, 'w') as f:
        f.writelines(lines)

    command = f'ffmpeg -i {mp4_file} -vf subtitles={srt_file} output_srt.mp4 -y'
    os.system(command)

# print(prediction)