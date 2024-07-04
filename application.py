import os
from openai import OpenAI
import tempfile
from datetime import date
from dotenv import load_dotenv
import whisper
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    results = model.transcribe(audio_path)
    return results["text"]

def split_text_into_chunks(text, chunk_size=3000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def abstract_summary_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    summaries = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a highly skilled AI trained in language comprehension and "
                                              "summarization. I would like you to read the following text and summarize"
                                              "it into a concise abstract paragraph. Aim to retain the most important "
                                              "points, providing a coherent and readable summary that could help a "
                                              "person understand the main points of the discussion without needing to "
                                              "read the entire text. Please avoid unnecessary details or tangential "
                                              "points."},
                {"role": "user", "content": chunk}
            ]
        )
        summaries.append(response.choices[0].message.content)

    return ' '.join(summaries)

def key_points_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    key_points = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a proficient AI with a specialty in distilling information into "
                                              "key points. Based on the following text, identify and list the main "
                                              "points that were discussed or brought up. These should be the most "
                                              "important ideas, findings, or topics that are crucial to the essence of "
                                              "the discussion. Your goal is to provide a list that someone could read "
                                              "to quickly understand what was talked about."},
                {"role": "user", "content": chunk}
            ]
        )
        key_points.append(response.choices[0].message.content)

    return ' '.join(key_points)

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    try:
        transcribed_text = transcribe_audio(audio_path)
        meeting_minutes_result = {
            'abstract_summary': abstract_summary_extraction(transcribed_text),
            'key_points': key_points_extraction(transcribed_text)
        }
        os.remove(audio_path)
        return jsonify(meeting_minutes_result)
    except Exception as e:
        return jsonify({'error': 'Error generating summary'}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=8000, debug=True)
