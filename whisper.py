import sounddevice as sd
import soundfile as sf
import openai
import os
import asyncio
from openai import OpenAI
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine, OpenAIEngine
import nltk

client = OpenAI(api_key='')

def convo_partner(tongue,rate,level):

    conversation_history = []

    def record_audio(file_name, duration=10, samplerate=44100):
        print("Recording...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        sf.write(file_name, audio_data, samplerate)
        print(f"Audio saved to {file_name}")


    while True:

        prompt = f"""You are a conversational partner to a user attempting to practice a foreign language at the 
        {level} level. You will continue a pleasant conversation in whichever language your user selected by asking 
        questions and giving statements related to the previous inputs by your conversational partner. 
        If they make mistakes, you will correct not outright, but by using the correct word or syntax in your next reply.
        If they are extremely beginner, DO NOT USE COMPLICATED PHRASES OR EXPRESSIONS, just stick to the very basics 
        and often repeat the words they use, introducing new concepts slowly and deliberately.
        I cannot repeat this enough, if they are extremely beginner, speak as if you are speaking to a YOUNG CHILD. 
        Still use correct language and grammar but do not use anything complicated. 
        Speak more simply than you think you should.
        After a few questions on the same topic, if you are repeating yourself, move to something different.
        Some acceptable topics might be pets, hobbies, television, weather, location, food, sports, or music.
        """
        
        # Example usage:
        record_audio("recorded_audio.wav")


        # Open the audio file
        audio_file = open("recorded_audio.wav", 'rb')

        # Create transcription using the Whisper model
        transcription = client.audio.transcriptions.create(
            model='whisper-1', 
            file=audio_file,
            language = tongue
        )

        # Print the transcription text
        print(transcription.text)

        # Close the audio file
        audio_file.close()

        os.remove("recorded_audio.wav")

        # Prompt the user for input
        user_input = transcription.text

        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Generate a response using ChatGPT
        response = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": prompt},
                *conversation_history  # Include conversation history as messages
            ]
        )

        # Extract and print the assistant's response
        assistant_response = response.choices[0].message.content
        print(assistant_response)

        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_response})

        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=assistant_response,
            speed = rate
        )

        # Save the audio data to a WAV file
        with open("output.wav", "wb") as f:
            f.write(response.content)

        # Read the WAV file
        audio_data, samplerate = sf.read("output.wav")

        # Play the audio
        sd.play(audio_data, samplerate)
        sd.wait()

        # Delete the WAV file
        os.remove("output.wav")

