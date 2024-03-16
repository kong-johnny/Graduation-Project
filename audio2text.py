import whisper

model = whisper.load_model("medium")
results = whisper.transcribe(model, "audio.wav")
print(results)
