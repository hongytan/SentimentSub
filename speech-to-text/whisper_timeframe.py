import stable_whisper

model = stable_whisper.load_model('base')
# this modified model run just like the original model but accepts additional arguments
result = model.transcribe('./audio-test-files/insane.wav')

result.to_srt_vtt('audio.srt')
result.to_ass('audio.ass')
# word_level=False : use only segment timestamps (i.e without the green highlight)
# segment_level=False : use only word timestamps

result.save_as_json('audio.json')
# save inference result for later processing