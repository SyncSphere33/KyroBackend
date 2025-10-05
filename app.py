# ----------------------------
# SyncSphere Flask Backend (Refactored & Cleaned)
# ----------------------------

# ----------------------------
# Standard Library Imports
# ----------------------------
import os
import uuid
import logging
import tempfile
from io import BytesIO

# ----------------------------
# Third-Party Imports (Lightweight Only)
# ----------------------------
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms as T

# ----------------------------
# Configuration / Constants
# ----------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
EDITED_FOLDER = "edited"
TRIMMED_FOLDER = "trimmed_videos"
CONVERTED_FOLDER = "converted_videos"
DOCX_OUTPUT_FOLDER = "docx_outputs"
STUDY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "study")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "gif", "txt", "docx", "pdf"}
FONT_PATH = "arial.ttf"
UPSCALE_FACTOR = 4
MODEL_PATH = "espcn_x4_trained.pth"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, EDITED_FOLDER, TRIMMED_FOLDER, 
               CONVERTED_FOLDER, DOCX_OUTPUT_FOLDER, STUDY_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# ESPCN Model Definition
# ----------------------------
class ESPCN(nn.Module):
    """ESPCN Super-Resolution Model"""
    def __init__(self, scale_factor=4):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# Load trained model if available
model = ESPCN(scale_factor=UPSCALE_FACTOR).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    logging.info("ESPCN model loaded successfully.")
model.eval()

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Max 20MB upload

# ----------------------------
# Helper Functions
# ----------------------------
def allowed_file(filename: str) -> bool:
    """Check if uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file) -> str:
    """Save uploaded file with unique name"""
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

def cleanup_file(path: str):
    """Remove temporary file safely"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.warning(f"Failed to delete {path}: {e}")

def save_uploaded_video(file) -> str:
    """Save uploaded video with unique filename"""
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

def cleanup_video(path: str):
    """Remove temporary video file safely"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.warning(f"Failed to delete video {path}: {e}")

# ----------------------------
# Text Extraction Helper Functions
# ----------------------------
def extract_text(file):
    """Extract text from various file formats"""
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        import pdfplumber  # Lazy import
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    elif ext == "docx":
        from docx import Document  # Lazy import
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

def extract_text_for_summarizer(file):
    """Extract text from uploaded files for summarizer"""
    return extract_text(file)

def extract_text_for_qna(file):
    """Extract text from uploaded files for Q&A"""
    return extract_text(file)

# ----------------------------
# Study Tools Helper Functions
# ----------------------------
def generate_flashcards(text, max_cards=20):
    """
    Generate simple flashcards from text
    Args:
        text: string
        max_cards: maximum number of flashcards
    Returns:
        list of dicts [{question, answer}]
    """
    from nltk.tokenize import sent_tokenize  # Lazy import
    sentences = sent_tokenize(text)
    flashcards = []
    for i, sent in enumerate(sentences[:max_cards]):
        question = f"Q{i+1}: Explain this?"
        answer = sent.strip()
        flashcards.append({"question": question, "answer": answer})
    return flashcards

def summarize_text(text, length_option="Short (1 Paragraph)"):
    """Summarize text based on length option"""
    from nltk.tokenize import sent_tokenize  # Lazy import
    sentences = sent_tokenize(text)
    if length_option.startswith("Very Short"):
        n = min(3, len(sentences))
    elif length_option.startswith("Short"):
        n = min(5, len(sentences))
    elif length_option.startswith("Medium"):
        n = min(10, len(sentences))
    else:
        n = len(sentences)
    return " ".join(sentences[:n]).strip()

# ----------------------------
# Image APIs
# ----------------------------
@app.route("/api/image/remove-bg", methods=["POST"])
def remove_bg():
    """Remove background from image using rembg"""
    try:
        from rembg import remove  # Lazy import
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        input_path = save_uploaded_file(file)
        output_path = os.path.join(OUTPUT_FOLDER, f"bg_removed_{file.filename}")

        with open(input_path, "rb") as i:
            output_data = remove(i.read())
        with open(output_path, "wb") as o:
            o.write(output_data)

        cleanup_file(input_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/enhance", methods=["POST"])
def enhance_image():
    """
    Enhance an uploaded image using the ESPCN (Efficient Sub-Pixel
    Convolutional Neural Network) super-resolution model.
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        img = Image.open(file).convert("RGB")

        max_size = 1024
        if max(img.width, img.height) > max_size:
            scale = max_size / max(img.width, img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.BICUBIC)

        to_tensor = T.ToTensor()
        img_tensor = to_tensor(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output_tensor = model(img_tensor)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        enhanced_img = T.ToPILImage()(output_tensor.squeeze(0).cpu())

        buf = BytesIO()
        enhanced_img.save(buf, "PNG")
        buf.seek(0)

        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"enhanced_{secure_filename(file.filename)}"
        )

    except RuntimeError as e:
        logging.error(f"RuntimeError: {str(e)}")
        return jsonify({"error": "Memory error during enhancement. Try using a smaller image."}), 500

    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    """Generate image using Pollinations AI"""
    try:
        import requests  # Lazy import
        from urllib.parse import quote  # Lazy import
        import io  # Lazy import
        
        data = request.get_json()
        print("Request JSON:", data)

        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        response = requests.get(url)
        print("Pollinations API status:", response.status_code)

        if not response.ok:
            return jsonify({"error": "Failed to generate image"}), 502

        return send_file(io.BytesIO(response.content), mimetype="image/jpeg", as_attachment=False)

    except Exception as e:
        print("Exception:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/extend-bg", methods=["POST"])
def extend_bg():
    """Add padding around image background"""
    try:
        file = request.files.get('file')
        padding = int(request.form.get("padding", 50))
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        img = Image.open(file).convert("RGB")
        new_img = Image.new("RGB", (img.width + 2*padding, img.height + 2*padding), (255,255,255))
        new_img.paste(img, (padding, padding))

        buf = BytesIO()
        new_img.save(buf, "PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True,
                         download_name=f"extended_{file.filename}")
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/compress", methods=["POST"])
def compress_image():
    """Compress image to reduce file size"""
    try:
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        img = Image.open(file)
        output_path = os.path.join(OUTPUT_FOLDER, f"compressed_{file.filename}")
        img.save(output_path, optimize=True, quality=50)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/convert", methods=["POST"])
def convert_image_format():
    """Convert image to a different format"""
    try:
        file = request.files.get("file")
        target_format = request.form.get("format")
        if not file or not allowed_file(file.filename) or not target_format:
            return jsonify({"error": "Invalid file or format"}), 400

        img = Image.open(file)
        output_path = os.path.join(OUTPUT_FOLDER, f"converted_image.{target_format}")
        img.save(output_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/watermark", methods=["POST"])
def add_watermark():
    """Add text watermark to image"""
    try:
        file = request.files.get("file")
        watermark_text = request.form.get("watermark")
        if not file or not watermark_text:
            return jsonify({"error": "File and watermark text required"}), 400

        img = Image.open(file).convert("RGBA")
        watermark = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)

        font_size = max(20, int(img.width / 20))
        font = ImageFont.truetype(FONT_PATH, font_size) if os.path.exists(FONT_PATH) else ImageFont.load_default()

        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        position = (img.width - text_width - 10, img.height - text_height - 10)
        draw.text(position, watermark_text, fill=(255, 255, 255, 128), font=font)

        watermarked = Image.alpha_composite(img, watermark)
        output_path = os.path.join(OUTPUT_FOLDER, f"watermarked_{file.filename}")
        watermarked.convert("RGB").save(output_path)
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        logging.exception("Watermark processing failed:")
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/to_pdf", methods=["POST"])
def images_to_pdf():
    """Convert multiple images to a single PDF"""
    try:
        files = request.files.getlist("files")
        if not files or len(files) == 0:
            return jsonify({"error": "No files uploaded"}), 400

        images = []
        for f in files:
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

            img = Image.open(file_path).convert("RGB")
            images.append(img)

        if len(images) == 0:
            return jsonify({"error": "No valid images"}), 400

        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

        images[0].save(pdf_path, save_all=True, append_images=images[1:])

        return send_file(pdf_path, as_attachment=True, download_name="converted.pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Video APIs
# ----------------------------
@app.route("/api/video/trim", methods=["POST"])
def trim_video():
    """Trim a video to a specified start and end time (in seconds)"""
    try:
        from moviepy.editor import VideoFileClip  # Lazy import
        file = request.files.get("file")
        start = float(request.form.get("start", 0))
        end = float(request.form.get("end", 0))
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)
        end = min(end, clip.duration) if end > 0 else clip.duration
        trimmed_clip = clip.subclip(start, end)

        output_path = os.path.join(TRIMMED_FOLDER, f"trimmed_{file.filename}")
        trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()
        trimmed_clip.close()
        cleanup_video(input_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/video/img-to-video", methods=["POST"])
def img_to_video():
    """
    Accept multiple images (any extension) and convert them into a video.
    Form Data:
        - images: multiple image files
        - fps: optional, frames per second (default 1)
    Returns:
        - MP4 video file
    """
    try:
        import cv2  # Lazy import
        
        images = request.files.getlist("images")
        if not images:
            return jsonify({"error": "No images uploaded"}), 400

        fps = int(request.form.get("fps", 1))
        temp_dir = tempfile.mkdtemp()

        img_paths = []
        for img_file in images:
            filename = secure_filename(img_file.filename)
            path = os.path.join(temp_dir, filename)
            img_file.save(path)
            img_paths.append(path)

        img_paths.sort()

        first_img = cv2.imread(img_paths[0])
        height, width, layers = first_img.shape

        video_filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join(temp_dir, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            video.write(img)

        video.release()

        return send_file(video_path, as_attachment=True, download_name="output_video.mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/video/convert", methods=["POST"])
def convert_video():
    """Convert video to different formats"""
    try:
        from moviepy.editor import VideoFileClip  # Lazy import
        file = request.files.get("file")
        target_format = request.form.get("format", "mp4").lower()
        if not file or not target_format:
            return jsonify({"error": "File and target format required"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)

        base_name = os.path.splitext(file.filename)[0]
        output_filename = f"{base_name}.{target_format}"
        output_path = os.path.join(CONVERTED_FOLDER, output_filename)

        if target_format == "gif":
            clip.write_gif(output_path)
        elif target_format in ["mp3", "wav", "aac"]:
            if clip.audio:
                clip.audio.write_audiofile(output_path)
            else:
                clip.close()
                cleanup_video(input_path)
                return jsonify({"error": "No audio track in video"}), 400
        else:
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        clip.close()
        cleanup_video(input_path)
        return send_file(output_path, as_attachment=True, download_name=output_filename)

    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/video/adjust-speed", methods=["POST"])
def adjust_video_speed():
    """Change video speed (fast/slow)"""
    try:
        from moviepy.editor import VideoFileClip
        from moviepy.video.fx.all import speedx  # Lazy import
        file = request.files.get("file")
        factor = float(request.form.get("factor", 1))
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)
        modified_clip = speedx(clip, factor=factor)

        output_path = os.path.join(CONVERTED_FOLDER, f"speed_{file.filename}")
        modified_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()
        modified_clip.close()
        cleanup_video(input_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/video/merge", methods=["POST"])
def merge_videos():
    """Merge multiple videos into one"""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # Lazy import
        files = request.files.getlist("files")
        if not files or len(files) < 2:
            return jsonify({"error": "At least 2 videos required"}), 400

        clips = []
        temp_paths = []
        for file in files:
            path = save_uploaded_video(file)
            temp_paths.append(path)
            clips.append(VideoFileClip(path))

        final_clip = concatenate_videoclips(clips)
        output_path = os.path.join(OUTPUT_FOLDER, f"merged_{uuid.uuid4().hex}.mp4")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        for clip in clips:
            clip.close()
        for path in temp_paths:
            cleanup_video(path)

        final_clip.close()
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/video/extract-frames", methods=["POST"])
def extract_frames():
    """Extract frames from video at specified intervals"""
    try:
        from moviepy.editor import VideoFileClip  # Lazy import
        import zipfile  # Lazy import
        
        file = request.files.get("file")
        interval = float(request.form.get("interval", 1))
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)

        zip_name = f"{uuid.uuid4().hex}_frames.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, t in enumerate(range(0, int(clip.duration), int(interval))):
                frame = clip.to_ImageClip(t).img
                img = Image.fromarray(frame)
                img_filename = f"frame_{i+1}.png"
                img_path = os.path.join(OUTPUT_FOLDER, img_filename)
                img.save(img_path)
                zipf.write(img_path, arcname=img_filename)
                os.remove(img_path)

        clip.close()
        cleanup_video(input_path)
        return send_file(zip_path, as_attachment=True, download_name="frames.zip")

    except Exception as e:
        logging.error(e, exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----------------------------
# PDF APIs
# ----------------------------
@app.route("/api/pdf/to_images", methods=["POST"])
def pdf_to_images():
    """Convert PDF pages to images"""
    try:
        import zipfile  # Lazy import
        import fitz  # PyMuPDF - Lazy import
        
        files = request.files.getlist("file")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        zip_name = f"{uuid.uuid4()}.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_name)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files:
                filename = file.filename
                pdf_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(pdf_path)

                doc = fitz.open(pdf_path)
                folder_name = filename.replace(".pdf", "")
                folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                for i, page in enumerate(doc):
                    img = page.get_pixmap()
                    img_path = os.path.join(folder_path, f"{i+1}.png")
                    img.save(img_path)
                    zipf.write(img_path, arcname=f"{folder_name}/{i+1}.png")

        return send_file(zip_path, as_attachment=True, download_name="images_folder.zip")
    
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Study Tools APIs
# ----------------------------
@app.route("/api/study/summarize", methods=["POST"])
def generate_summary():
    """Generate text summary from uploaded document"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        length_option = request.form.get("length", "Short (1 Paragraph)")
        text = extract_text_for_summarizer(file)
        summary = summarize_text(text, length_option)

        return summary, 200, {"Content-Type": "text/plain"}

    except Exception as e:
        logging.error(f"Summarizer failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/study/flashcards", methods=["POST"])
def flashcards_api():
    """Generate flashcards from uploaded document"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        flashcards = generate_flashcards(text)
        return jsonify({"flashcards": flashcards})
    except Exception as e:
        logging.error(f"Flashcards generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/study/qna", methods=["POST"])
def generate_qna():
    """Generate Q&A pairs from uploaded document"""
    try:
        from nltk.tokenize import sent_tokenize  # Lazy import
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        difficulty = request.form.get("difficulty", "Intermediate")
        count = int(request.form.get("count", 10))

        text = extract_text_for_qna(file)
        sentences = sent_tokenize(text)
        sentences = sentences[:count]

        qna_list = [
            {
                "question": f"Q{i+1}: Can you explain this?",
                "answer": sent.strip()
            } for i, sent in enumerate(sentences)
        ]

        return jsonify({"qna": qna_list})

    except Exception as e:
        logging.error(f"Q&A generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/study/keywords", methods=["POST"])
def extract_keywords():
    """Extract keywords from uploaded document"""
    try:
        from nltk.tokenize import word_tokenize  # Lazy import
        import nltk  # Lazy import
        
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        words = word_tokenize(text)
        freq_dist = nltk.FreqDist(words)
        keywords = [word for word, _ in freq_dist.most_common(10)]
        return jsonify({"keywords": keywords})
    except Exception as e:
        logging.error(f"Keyword extraction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/study/sentiment", methods=["POST"])
def analyze_sentiment():
    """Analyze sentiment of sentences in uploaded document"""
    try:
        from nltk.tokenize import sent_tokenize  # Lazy import
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        sentences = sent_tokenize(text)
        sentiments = [{"sentence": sent, "sentiment": "positive" if "good" in sent.lower() else "negative"} for sent in sentences]
        return jsonify({"sentiments": sentiments})
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/study/paraphrase", methods=["POST"])
def paraphrase_text():
    """Paraphrase text from uploaded document"""
    try:
        from nltk.tokenize import sent_tokenize  # Lazy import
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        sentences = sent_tokenize(text)
        paraphrased = [f"Rewritten: {sent}" for sent in sentences]
        return jsonify({"paraphrased": paraphrased})
    except Exception as e:
        logging.error(f"Paraphrasing failed: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Frontend Routes
# ----------------------------
@app.route("/")
def home():
    """Serve home page"""
    return send_file("index.html")

@app.route("/study.html")
def study():
    """Serve study page"""
    return send_file("study.html")

@app.route("/tools.html")
def tools():
    """Serve tools page"""
    return send_file("tools.html")

# ----------------------------
# Run Flask Server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
