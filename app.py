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
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')  # base, small, medium, large

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load Whisper model on startup
logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
try:
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(file_path):
    """Safely delete a temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")

def transcribe_audio(file_path):
    """Transcribe audio file using Whisper."""
    try:
        if model is None:
            return {
                "success": False,
                "error": "Whisper model not loaded"
            }
        
        logger.info(f"Starting transcription for: {file_path}")
        result = model.transcribe(file_path)
        
        return {
            "success": True,
            "transcription": result["text"].strip(),
            "language": result.get("language", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment platforms."""
    return jsonify({
        "status": "healthy",
        "whisper_model": WHISPER_MODEL,
        "model_loaded": model is not None
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
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
    """Handle POST request with audio file for transcription."""
    temp_file_path = None
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False,
                "error": "Transcription service unavailable"
            }), 503
        
        # Check if file is present in request
        if 'audio' not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file provided. Use 'audio' as the form field name."
            }), 400
        
        file = request.files['audio']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not supported. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Create secure filename
        filename = secure_filename(file.filename)
        if not filename:
            filename = "audio_file"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
            logger.info(f"Saved uploaded file to: {temp_file_path}")
        
        # Get file size for logging
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Processing file: {filename}, Size: {file_size} bytes")
        
        # Transcribe audio
        result = transcribe_audio(temp_file_path)
        
        if result["success"]:
            logger.info("Transcription completed successfully")
            return jsonify({
                "success": True,
                "transcription": result["transcription"],
                "language": result.get("language", "unknown"),
                "filename": filename,
                "file_size_bytes": file_size
            })
        else:
            logger.error(f"Transcription failed: {result['error']}")
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in transcribe endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Internal server error occurred"
        }), 500
    
    finally:
        # Always cleanup temporary file
        if temp_file_path:
            cleanup_file(temp_file_path)

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error."""
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size allowed: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB"
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Get port from environment variable (required for Render and similar platforms)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )