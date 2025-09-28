import os
import tempfile
import logging
from pathlib import Path
import whisper
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB max file size
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'tiny')  # Use 'tiny' to reduce memory usage

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load Whisper model
logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
try:
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")

def transcribe_audio(file_path):
    try:
        if model is None:
            return {"success": False, "error": "Whisper model not loaded"}
        logger.info(f"Transcribing: {file_path}")
        result = model.transcribe(file_path)
        return {
            "success": True,
            "transcription": result["text"].strip(),
            "language": result.get("language", "unknown")
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"success": False, "error": f"Transcription failed: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "whisper_model": WHISPER_MODEL,
        "model_loaded": model is not None
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Flask Audio Transcription Server",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Upload audio file for transcription",
            "/health": "GET - Health check"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
        "whisper_model": WHISPER_MODEL
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    temp_file_path = None
    try:
        if model is None:
            return jsonify({"success": False, "error": "Transcription service unavailable"}), 503

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided. Use 'file' as the form field name."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        filename = secure_filename(file.filename) or "audio_file"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
            logger.info(f"Saved file: {temp_file_path}")

        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File size: {file_size} bytes")

        result = transcribe_audio(temp_file_path)
        if result["success"]:
            return jsonify({
                "success": True,
                "transcription": result["transcription"],
                "language": result.get("language", "unknown"),
                "filename": filename,
                "file_size_bytes": file_size
            })
        else:
            return jsonify({"success": False, "error": result["error"]}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Internal server error"}), 500
    finally:
        if temp_file_path:
            cleanup_file(temp_file_path)

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": f"File too large. Max size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB"
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
