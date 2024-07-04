import os
from openai import OpenAI
import tempfile
from datetime import date
from dotenv import load_dotenv
import whisper
# import language_tool_python
# from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline
# import spacy
from flask import Flask, send_file, request, jsonify
import os
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

client = OpenAI(
     api_key = os.environ.get("OPENAI_API_KEY")
)
# openai_key = os.getenv("OPENAI_KEY")
# openai.api_key = openai_key
# audio_path = "./audio.mp3"

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# model_name = 'facebook/bart-large-cnn'
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)
# tool = language_tool_python.LanguageTool('en-US')
# nlp = spacy.load("en_core_web_sm")

# transcribing the audio file to raw text
def transcribe_audio(audio_path):
        model = whisper.load_model("base")
        results = model.transcribe(audio_path)
        # print(results["text"])
        return(results["text"])

# transcription = transcribe_audio("audio.mp3")

# def meeting_minutes(transcription):
#     abstract_summary = abstract_summary_extraction(transcription)
#     print(abstract_summary)
#     key_points = key_points_extraction(transcription)
#     print(key_points)
#     return {
#         'abstract_summary': abstract_summary,
#         'key_points': key_points,
#     }


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

# meeting_minutes(transcription);

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
