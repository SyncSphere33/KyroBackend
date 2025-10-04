# ----------------------------
# SYncSphere Flask Backend (Refactored & Cleaned)
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
# Third-Party Imports
# ----------------------------
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms as T
from rembg import remove
from docx import Document
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import speedx
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download("punkt")

# ----------------------------
# Configuration / Constants
# ----------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
EDITED_FOLDER = "edited"
TRIMMED_FOLDER = "trimmed_videos"
CONVERTED_FOLDER = "converted_videos"
DOCX_OUTPUT_FOLDER = "docx_outputs"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "gif", "txt", "docx", "pdf"}
FONT_PATH = "arial.ttf"
UPSCALE_FACTOR = 4
MODEL_PATH = "espcn_x4_trained.pth"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, EDITED_FOLDER, TRIMMED_FOLDER, CONVERTED_FOLDER]:
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
from flask_cors import CORS
CORS(app)
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

# ----------------------------
# ----------------------------
# Image APIs
# ----------------------------
@app.route("/api/image/remove-bg", methods=["POST"])
def remove_bg():
    """Remove background from image using rembg"""
    try:
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

# --------------------------------------------------------
# IMAGE ENHANCEMENT ROUTE – Using ESPCN Super-Resolution
# --------------------------------------------------------

@app.route("/api/image/enhance", methods=["POST"])
def enhance_image():
    """
    Enhance an uploaded image using the ESPCN (Efficient Sub-Pixel
    Convolutional Neural Network) super-resolution model.
    - Accepts a single image file (PNG/JPEG).
    - Resizes overly large images to prevent memory crashes.
    - Returns the enhanced high-resolution image as a downloadable PNG.
    """
    try:
        # 1️⃣ Retrieve uploaded file from the POST request
        file = request.files.get('file')

        # If no file is found in the request, return an error
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # 2️⃣ Open the uploaded file using Pillow (PIL) and ensure RGB mode
        img = Image.open(file).convert("RGB")

        # 3️⃣ Prevent excessive memory usage by scaling down very large images
        max_size = 1024  # Maximum width/height allowed (in pixels)
        if max(img.width, img.height) > max_size:
            # Calculate a scale factor to resize the image proportionally
            scale = max_size / max(img.width, img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # Resize the image using bicubic interpolation for better quality
            img = img.resize((new_width, new_height), Image.BICUBIC)

        # 4️⃣ Convert Pillow image to PyTorch tensor (values scaled to [0,1])
        to_tensor = T.ToTensor()           # Transform to tensor
        img_tensor = to_tensor(img)        # Shape: [C, H, W]
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension → [1, C, H, W]

        # Move tensor to GPU if available (device is set globally at startup)
        img_tensor = img_tensor.to(device)

        # 5️⃣ Perform super-resolution inference with ESPCN model
        with torch.no_grad():  # Disable gradient calculation (saves memory)
            output_tensor = model(img_tensor)     # Pass image through the model
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)  # Clip values to [0,1]

        # 6️⃣ Convert enhanced tensor back to a Pillow image for saving
        enhanced_img = T.ToPILImage()(output_tensor.squeeze(0).cpu())

        # 7️⃣ Save enhanced image into an in-memory buffer
        buf = BytesIO()
        enhanced_img.save(buf, "PNG")  # Save as PNG format
        buf.seek(0)  # Reset buffer position to the start

        # 8️⃣ Send the enhanced image back to the user as a downloadable file
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"enhanced_{secure_filename(file.filename)}"
        )

    except RuntimeError as e:
        # Handles cases like GPU/CPU out-of-memory errors
        logging.error(f"RuntimeError: {str(e)}")
        return jsonify({"error": "Memory error during enhancement. Try using a smaller image."}), 500

    except Exception as e:
        # Handles any unexpected errors
        logging.error(f"Exception: {str(e)}")
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

        # Use textbbox to get the text size
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

import uuid

# ----------------------------
# Images to PDF Converter
# ----------------------------
@app.route("/api/image/to_pdf", methods=["POST"])
def images_to_pdf():
    try:
        # Get all uploaded files (match 'files' from frontend FormData)
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

        # Create a unique PDF name
        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

        # Save all images into a single PDF
        images[0].save(pdf_path, save_all=True, append_images=images[1:])

        return send_file(pdf_path, as_attachment=True, download_name="converted.pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Video APIs
# ----------------------------
# Similar clean-up and docstrings for all video routes
# (trim, convert, enhance, generate, merge, extract-frames, adjust-speed)

# ----------------------------
# Video Helper Functions
# ----------------------------
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
# Video APIs
# ----------------------------
#trim video
@app.route("/api/video/trim", methods=["POST"])
def trim_video():
    """
    Trim a video to a specified start and end time (in seconds)
    Form Data:
    - file: video file
    - start: start time (float)
    - end: end time (float)
    """
    try:
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

import cv2
import tempfile


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
        images = request.files.getlist("images")
        if not images:
            return jsonify({"error": "No images uploaded"}), 400

        fps = int(request.form.get("fps", 1))  # default 1 fps

        # Create a temp directory to save images
        temp_dir = tempfile.mkdtemp()

        # Save images to temp folder and collect their paths
        img_paths = []
        for img_file in images:
            filename = secure_filename(img_file.filename)
            path = os.path.join(temp_dir, filename)
            img_file.save(path)
            img_paths.append(path)

        # Sort images by filename (optional, if user wants order)
        img_paths.sort()

        # Read first image to get dimensions
        first_img = cv2.imread(img_paths[0])
        height, width, layers = first_img.shape

        # Define video file path
        video_filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join(temp_dir, video_filename)

        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Add all images to video
        for img_path in img_paths:
            img = cv2.imread(img_path)

            # Resize image if dimensions don't match first image
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))

            video.write(img)

        video.release()

        # Send video as response
        return send_file(video_path, as_attachment=True, download_name="output_video.mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/convert", methods=["POST"])
def convert_video():
    try:
        file = request.files.get("file")
        target_format = request.form.get("format", "mp4").lower()  # get user selection
        if not file or not target_format:
            return jsonify({"error": "File and target format required"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)

        base_name = os.path.splitext(file.filename)[0]
        output_filename = f"{base_name}.{target_format}"
        output_path = os.path.join(CONVERTED_FOLDER, output_filename)

        # Choose conversion based on target_format
        if target_format == "gif":
            clip.write_gif(output_path)
        elif target_format in ["mp3", "wav", "aac"]:  # audio-only formats
            if clip.audio:
                clip.audio.write_audiofile(output_path)
            else:
                clip.close()
                cleanup_video(input_path)
                return jsonify({"error": "No audio track in video"}), 400
        else:  # video formats
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        clip.close()
        cleanup_video(input_path)
        return send_file(output_path, as_attachment=True, download_name=output_filename)

    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/adjust-speed", methods=["POST"])
def adjust_video_speed():
    """
    Change video speed (fast/slow)
    Form Data:
    - file: video file
    - factor: speed factor (>1 = faster, <1 = slower)
    """
    try:
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
    """
    Merge multiple videos into one
    Form Data:
    - files: multiple video files
    """
    try:
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

        # Close all clips and cleanup
        for clip in clips:
            clip.close()
        for path in temp_paths:
            cleanup_video(path)

        final_clip.close()
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

import logging
from moviepy.editor import VideoFileClip
import zipfile
from PIL import Image

def save_uploaded_video(file):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

def cleanup_video(path):
    if os.path.exists(path):
        os.remove(path)

@app.route("/api/video/extract-frames", methods=["POST"])
def extract_frames():
    try:
        file = request.files.get("file")
        interval = float(request.form.get("interval", 1))
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        input_path = save_uploaded_video(file)
        clip = VideoFileClip(input_path)

        # Prepare ZIP
        zip_name = f"{uuid.uuid4().hex}_frames.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Extract frames at specified interval
            for i, t in enumerate(range(0, int(clip.duration), int(interval))):
                frame = clip.to_ImageClip(t).img  # get frame at time t (seconds)
                img = Image.fromarray(frame)
                img_filename = f"frame_{i+1}.png"
                img_path = os.path.join(OUTPUT_FOLDER, img_filename)
                img.save(img_path)
                zipf.write(img_path, arcname=img_filename)
                os.remove(img_path)  # cleanup individual frame

        clip.close()
        cleanup_video(input_path)

        # Return ZIP file
        return send_file(zip_path, as_attachment=True, download_name="frames.zip")

    except Exception as e:
        logging.error(e, exc_info=True)
        return jsonify({"error": str(e)}), 500
# ----------------------------
# File APIs
# ----------------------------
# DOCX view, PDF to DOCX, split/merge, multi-img to PDF
# ----------------------------
# DOCX Helper Functions
# ----------------------------
def save_uploaded_docx(file) -> str:
    """Save uploaded DOCX with unique filename"""
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
        logging.warning(f"Failed to delete file {path}: {e}")


# ----------------------------
# DOCX APIs
# ----------------------------

@app.route("/api/docx/edit", methods=["POST"])
def edit_docx():
    """
    Read + Edit DOCX
    Form Data:
    - file: DOCX file
    - text: text to append or replace
    """
    try:
        file = request.files.get("file")
        text_to_add = request.form.get("text", "")

        if not file:
            return jsonify({"error": "No DOCX uploaded"}), 400

        input_path = save_uploaded_docx(file)
        doc = Document(input_path)

        # Optional: replace all text instead of appending
        if text_to_add:
            # Clear all existing paragraphs
            for para in doc.paragraphs:
                p = para._element
                p.getparent().remove(p)

            # Add new text
            doc.add_paragraph(text_to_add)

        output_filename = f"edited_{uuid.uuid4().hex}_{file.filename}"
        output_path = os.path.join(DOCX_OUTPUT_FOLDER, output_filename)
        doc.save(output_path)
        cleanup_file(input_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/docx/merge", methods=["POST"])
def merge_docx():
    """
    Merge multiple DOCX files into one
    Form Data:
    - files: multiple DOCX files
    """
    try:
        files = request.files.getlist("files")
        if not files or len(files) < 2:
            return jsonify({"error": "At least 2 DOCX files required"}), 400

        merged_doc = Document()
        temp_paths = []

        for file in files:
            path = save_uploaded_docx(file)
            temp_paths.append(path)
            doc = Document(path)
            for para in doc.paragraphs:
                merged_doc.add_paragraph(para.text)
        output_filename = f"merged_{uuid.uuid4().hex}.docx"
        output_path = os.path.join(DOCX_OUTPUT_FOLDER, output_filename)
        merged_doc.save(output_path)
        merged_doc.save(output_path)

        # Cleanup temporary files
        for path in temp_paths:
            cleanup_file(path)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/docx/split", methods=["POST"])
def split_docx():
    """
    Split DOCX into multiple files, each containing one paragraph
    Form Data:
    - file: DOCX file
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No DOCX uploaded"}), 400

        input_path = save_uploaded_docx(file)
        doc = Document(input_path)
        output_paths = []

        for i, para in enumerate(doc.paragraphs):
            new_doc = Document()
            new_doc.add_paragraph(para.text)
            output_filename = f"split_{i}_{uuid.uuid4().hex}.docx"
            output_path = os.path.join(DOCX_OUTPUT_FOLDER, output_filename)
            new_doc.save(output_path)
            output_paths.append(output_path)

        cleanup_file(input_path)
        return jsonify({"files": output_paths})
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

import zipfile
import fitz  

@app.route("/api/pdf/to_images", methods=["POST"])
def pdf_to_images():
    files = request.files.getlist("file")
    if not files:
        return {"error": "No files uploaded"}, 400

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
# ----------------------------
# Study Tools APIs
# ----------------------------
# Summarize, QnA, flashcards, keywords, sentiment, paraphrase
# ----------------------------
# Study Tools Helper Functions
STUDY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "study")
os.makedirs(STUDY_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOCX_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STUDY_OUTPUT_FOLDER, exist_ok=True)


def generate_flashcards(text, max_cards=20):
    """
    Generate simple flashcards from text
    Args:
        text: string
        max_cards: maximum number of flashcards
    Returns:
        list of dicts [{question, answer}]
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    flashcards = []
    for i, sent in enumerate(sentences[:max_cards]):
        question = f"Q{i+1}: Explain this?"
        answer = sent.strip()
        flashcards.append({"question": question, "answer": answer})
    return flashcards




# ----------------------------
# Summarize Notes API
# ----------------------------
# 1. Notes Summarizer (NLTK-based fallback)
# ----------------------------
# Notes Summarizer API
# ----------------------------


from nltk.tokenize import sent_tokenize
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Max 20MB upload

# Helper function to extract text from uploaded files
# Extract text from file
def extract_text_for_summarizer(file):
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    elif ext == "docx":
        from docx import Document
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")
# Simple summarizer function
def summarize_text(text, length_option="Short (1 Paragraph)"):
    sentences = sent_tokenize(text)
    if length_option.startswith("Very Short"):
        n = min(3, len(sentences))
    elif length_option.startswith("Short"):
        n = min(5, len(sentences))
    elif length_option.startswith("Medium"):
        n = min(10, len(sentences))
    else:  # Detailed
        n = len(sentences)
    return " ".join(sentences[:n]).strip()

@app.route("/api/study/summarize", methods=["POST"])
def generate_summary():
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
# Flashcards API
# ----------------------------
@app.route("/api/study/flashcards", methods=["POST"])
def flashcards_api():
    """
    Generate flashcards from uploaded text-based files
    Form Data:
    - file: uploaded file
    """
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


# ----------------------------
# Q&A API
# ----------------------------
# Q&A API
# ----------------------------
from flask import Flask, request, jsonify

from nltk.tokenize import sent_tokenize

# Utility function to extract text from uploaded file
def extract_text_for_qna(file):
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        import tempfile, os, docx2txt
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    else:
        raise ValueError("Unsupported file type")
@app.route("/api/study/qna", methods=["POST"])
def generate_qna():
    """
    Generate simple Q&A from uploaded text-based files
    Form Data:
    - file: uploaded file
    - difficulty: optional (Basic/Intermediate/Advanced)
    - count: optional number of questions
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Optional parameters from frontend
        difficulty = request.form.get("difficulty", "Intermediate")
        count = int(request.form.get("count", 10))

        text = extract_text_for_qna(file)
        sentences = sent_tokenize(text)

        # Limit the number of questions
        sentences = sentences[:count]

        # Simple Q&A generation
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

# ----------------------------
# Keyword Extraction API
# ----------------------------
@app.route("/api/study/keywords", methods=["POST"])
def extract_keywords():
    """
    Extract top 10 frequent words as keywords from uploaded text-based files
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        from nltk.tokenize import word_tokenize
        words = word_tokenize(text)
        freq_dist = nltk.FreqDist(words)
        keywords = [word for word, _ in freq_dist.most_common(10)]
        return jsonify({"keywords": keywords})
    except Exception as e:
        logging.error(f"Keyword extraction failed: {e}")
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Sentiment Analysis API
# ----------------------------
@app.route("/api/study/sentiment", methods=["POST"])
def analyze_sentiment():
    """
    Perform simple sentiment analysis on uploaded text-based files
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        sentiments = [{"sentence": sent, "sentiment": "positive" if "good" in sent.lower() else "negative"} for sent in sentences]
        return jsonify({"sentiments": sentiments})
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Paraphrasing API
# ----------------------------
@app.route("/api/study/paraphrase", methods=["POST"])
def paraphrase_text():
    """
    Perform simple paraphrasing by rewriting sentences
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        text = extract_text(file)
        from nltk.tokenize import sent_tokenize
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets the PORT automatically
    app.run(host="0.0.0.0", port=port)

