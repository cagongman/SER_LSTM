from openai import OpenAI
client = OpenAI(api_key='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')

audio_file = open("./audio/myAudio.wav", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

print('input text: ' + transcript.text)


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an emotion analyst who analyzes the input text and recognizes emotions like happy, sad, angry, and neutral."},
    {"role": "user", "content": "Hello my name is kisub"},
    {"role": "assistant", "content": "Your emotional state right now is neutral"},
    {"role": "user", "content": "I don't know... I just feel so tired all the time."},
    {"role": "assistant", "content": "Your emotional state right now is sad."},
    {"role": "user", "content": "It's all your fault! You always mess things up."},
    {"role": "assistant", "content": "Your emotional state right now is angry."},
    {"role": "user", "content": transcript.text}
  ]
)

print(completion.choices[0].message)
