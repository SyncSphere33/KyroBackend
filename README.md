# 🌟 Kyro – “Keep Your Resources Organized” – AI-Powered Productivity & Study Toolkit

Kyro – “Keep Your Resources Organized” is your **all-in-one study companion**.
It combines **image, document, and video tools** with **AI-powered study helpers** like summarization, Q&A generation, flashcards, keyword extraction, and more — all in a **simple web interface** built with **Flask (Python backend) + HTML/CSS/JS frontend**.

---

## 🚀 Why It Matters

We all face:

* 📚 **Information overload:** Hard to digest huge notes and PDFs → our **Summarizer & Keyword Extractor** help you focus on key points.
* ⚡ **Productivity challenges:** Need to trim videos, convert docs, remove image backgrounds → all done in one place.
* 🧠 **Personalized learning:** Our **AI tools** create Q&As, flashcards, and summaries tailored to you.
* 💻 **Too many apps:** Instead of juggling 5-6 apps, this single platform handles everything.

---

## ✨ Main Features

### 🖼️ Image Tools

* Remove background (`rembg`)
* Compress & convert formats (JPG ↔ PNG ↔ WebP)
* AI **super-resolution enhancer** (clearer images)
* Add watermark / extend background
* Convert images → PDF

### 📄 Document Tools

* Merge / split / edit **DOCX**
* PDF → Images / PDF → DOCX
* View DOCX online

### 🎬 Video Tools

* Trim, merge, convert formats
* Adjust playback speed
* Extract frames
* Create slideshow videos from images

### 🤖 AI Study Tools

* Text **Summarizer**
* **Q&A Generator** (from notes/PDFs)
* **Flashcard Creator**
* **Keyword Extractor**
* **Paraphraser & Sentiment Analyzer**

---

## 🖥️ Tech Stack

* **Backend:** Python + Flask
* **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript
* **AI/NLP:** `nltk`, `sumy`, `torch`, `rembg`
* **Multimedia:** `moviepy`, `opencv-python`, `Pillow`
* **Documents:** `python-docx`, `pdfplumber`, `PyMuPDF`, `PyPDF2`

---

## ⚙️ Requirements

* Python **3.8 or higher** (recommended: 3.10+)
* OS: Windows / macOS / Linux
* Browser: Chrome, Firefox, or Edge

---

## 📦 Key Python Libraries (and WHY they’re needed)

| Library                 | Why We Need It                                                          |
| ----------------------- | ----------------------------------------------------------------------- |
| **Flask**               | Runs the backend web server and connects the tools to the browser.      |
| **Pillow**              | For opening, editing, and saving images (e.g., compression, watermark). |
| **rembg**               | Removes the background from images using AI.                            |
| **python-docx**         | Reads and edits **.docx** Word documents.                               |
| **pdfplumber**          | Extracts text and images from PDFs for summarization/Q&A.               |
| **PyPDF2**              | Splits and merges PDF files.                                            |
| **PyMuPDF (fitz)**      | Converts PDFs to images and vice versa.                                 |
| **moviepy**             | For cutting, merging, and converting videos.                            |
| **opencv-python**       | Handles video/image processing and filters.                             |
| **nltk**                | Natural Language Toolkit → tokenization & AI text processing.           |
| **sumy**                | AI summarization engine.                                                |
| **torch + torchvision** | For advanced AI models (e.g., image enhancer).                          |
| **docx2txt**            | Extracts raw text from DOCX for the summarizer/Q&A generator.           |
| **fonttools**           | Used when adding watermarks or custom fonts.                            |
| **tqdm**                | Displays progress bars during long operations (like video processing).  |
| **Werkzeug**            | Flask’s internal helper for routing & file uploads.                     |

👉 **Note:** You only need these — no extra libraries unless you add more features.

---

## ⚡ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/smart-study-buddy-pro.git
cd smart-study-buddy-pro
```

### 2️⃣ Create a Virtual Environment

(Recommended so your system’s Python stays clean.)

```bash
python -m venv venv
```

* **Windows:**

```bash
venv\Scripts\activate
```

* **macOS/Linux:**

```bash
source venv/bin/activate
```

💡 *You’ll see `(venv)` at the start of your terminal when it’s active.*

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, create it:

```bash
pip freeze > requirements.txt
```

Or manually install only the **main packages**:


```bash
pip install flask==3.1.2 pillow==11.3.0 rembg==2.0.67 python-docx==1.2.0 pdfplumber==0.11.7 PyPDF2==3.0.1 pymupdf==1.26.4 moviepy==1.0.3 opencv-python==4.8.0.76 nltk==3.9.1 sumy==0.11.0 docx2txt==0.9 torch==2.8.0 torchvision==0.23.0 fonttools==4.60.0 tqdm==4.67.1 Werkzeug==3.1.3 onnxruntime numpy<2
```


👉 For **NLTK data** (needed for summarizer), run once:

```python
import nltk
nltk.download('punkt')
```

---

## ▶️ Running the Project

### 1. Start the Backend

```bash
python app.py
```

✅ Flask server runs at: `http://127.0.0.1:5000/`

### 2. Open the Web App

Go to your browser:

* **Home:** `http://localhost:5000/`
* **Tools:** `http://localhost:5000/tools.html`
* **AI Study:** `http://localhost:5000/study.html`

---

## 🗂️ Project Structure

```
smart-study-buddy-pro/
│
├── app.py               # Main Flask server
├── index.html           # Homepage
├── tools.html           # All productivity tools
├── study.html           # AI study tools
├── static/              # CSS, JS, icons
├── uploads/             # Auto-created: uploaded files
├── outputs/             # Auto-created: processed files
├── docx_outputs/        # Auto-created: DOCX exports
├── requirements.txt     # Library list
└── README.md            # This guide
```

---

## 🔧 Troubleshooting Tips

* 🖼️ **Missing Fonts:** If watermarking fails, ensure `arial.ttf` is in your project or update the font path in `app.py`.
* 📁 **Large Files:** Default upload size limit: **20MB**. Adjust `MAX_CONTENT_LENGTH` in `app.py` if needed.
* ⚙️ **Model Weights:** For the AI Image Enhancer, place the file `espcn_x4_trained.pth` in the root folder.

---

## ❤️ Acknowledgements

* [Flask](https://flask.palletsprojects.com/) – web framework
* [Pillow](https://python-pillow.org/) – image processing
* [rembg](https://github.com/danielgatis/rembg) – AI background remover
* [moviepy](https://zulko.github.io/moviepy/) – video editing
* [nltk](https://www.nltk.org/) & [sumy](https://github.com/miso-belica/sumy) – AI text tools
* [Tailwind CSS](https://tailwindcss.com/) & [Feather Icons](https://feathericons.com/) – modern UI

---

## 📜 License

Released under **MIT License** → free to use, modify, and distribute.

---

## 📬 Contact

For questions or suggestions: open an **Issue** on GitHub or email the maintainer.

---

### ✅ Beginner Takeaways:

1. **Always activate your virtual environment** before installing/running.
2. **Install only the listed packages** → they match our code exactly.
3. Each library has a **specific purpose** (see table above) → don’t install unnecessary extras.
4. If anything breaks, check **Troubleshooting** above first.

-