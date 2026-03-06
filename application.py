import os
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac', 'webm'}
MAX_FILE_SIZE_MB = 100

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    results = model.transcribe(audio_path)
    return results["text"]

def split_text_into_chunks(text, chunk_size=3000, overlap=200):
    """Split text into chunks with overlap to avoid cutting mid-sentence."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap to preserve context across chunks
    return chunks

def summarize_chunk(chunk):
    """Extract abstract summary for a single chunk."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "You are a highly skilled AI trained in language comprehension and summarization. "
                    "Read the following text and summarize it into a concise abstract paragraph. "
                    "Retain the most important points, providing a coherent and readable summary that "
                    "could help a person understand the main points without reading the entire text. "
                    "Avoid unnecessary details or tangential points."
                )},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarize_chunk: {e}")
        return "Summary extraction failed for this segment."

def extract_key_points_chunk(chunk):
    """Extract key points for a single chunk."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "You are a proficient AI with a specialty in distilling information into key points. "
                    "Based on the following text, identify and list the main points that were discussed. "
                    "These should be the most important ideas, findings, or topics crucial to the essence "
                    "of the discussion. Provide a list someone could read to quickly understand what was talked about."
                )},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in extract_key_points_chunk: {e}")
        return "Key points extraction failed for this segment."

def abstract_summary_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_chunk, chunks))
    return ' '.join(summaries)

def key_points_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        key_points = list(executor.map(extract_key_points_chunk, chunks))
    return ' '.join(key_points)

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if not audio_file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Check file size
    audio_file.seek(0, 2)  # seek to end
    file_size_mb = audio_file.tell() / (1024 * 1024)
    audio_file.seek(0)  # reset
    if file_size_mb > MAX_FILE_SIZE_MB:
        return jsonify({'error': f'File too large. Max size is {MAX_FILE_SIZE_MB}MB'}), 400

    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    try:
        transcribed_text = transcribe_audio(audio_path)

        # Run summary and key points extraction in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            summary_future = executor.submit(abstract_summary_extraction, transcribed_text)
            key_points_future = executor.submit(key_points_extraction, transcribed_text)
            abstract_summary = summary_future.result()
            key_points = key_points_future.result()

        return jsonify({
            'abstract_summary': abstract_summary,
            'key_points': key_points,
            'transcript': transcribed_text
        })
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({'error': 'Error generating summary'}), 500
    finally:
        # Always clean up the uploaded file
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=8000, debug=True)