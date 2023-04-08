import stable_whisper

model = stable_whisper.load_model('base')
# this modified model run just like the original model but accepts additional arguments
videofile = "/Users/hongtan/Desktop/sentimentsub/website/sentsub/media/videos/Friends_Joeys_Bad_Birthday_Gift.mp4"
result = model.transcribe(videofile, fp16=False)

result.to_srt_vtt('audio.srt', word_level=False)
result.to_ass('audio.ass')
#word_level=False : use only segment timestamps (i.e without the green highlight)
#segment_level=False : use only word timestamps

# result.save_as_json('audio.json')
# save inference result for later processing