"""
Helper utilities for the smart assistant.
"""
import os
import json
import base64
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from typing import Optional


def allowed_chat_file(filename, for_images=True):
    """Check if file is allowed in chat upload."""
    from config import config
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if for_images:
        return ext in config.CHAT_ALLOWED_IMAGE_EXT
    return ext in config.CHAT_ALLOWED_IMAGE_EXT or ext in config.CHAT_ALLOWED_TEXT_EXT


def _read_docx(path: str) -> str:
    """从 .docx 提取正文（docx 本质是 zip，正文在 word/document.xml）。"""
    import zipfile, re
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    # 每个 <w:t>...</w:t> 是一段文字，按段落 <w:p> 加换行
    paras = re.split(r"</w:p>", xml)
    out = []
    for para in paras:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S)
        line = "".join(texts)
        if line.strip():
            out.append(line)
    return "\n".join(out)


def read_text_file_content(path: str) -> Optional[str]:
    """Read the text content of a file for chat context."""
    try:
        if path.lower().endswith(".docx"):
            content = _read_docx(path)
            from config import config
            max_chars = config.CHAT_MAX_FILE_SIZE_MB * 50000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[文件内容已截断]..."
            return content
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            # Limit to reasonable length for LLM context
            from config import config
            max_chars = config.CHAT_MAX_FILE_SIZE_MB * 50000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[文件内容已截断]..."
            return content
    except Exception:
        return None


def image_to_base64(path: str) -> Optional[str]:
    """Read an image file and return a data URI string."""
    try:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                    "gif": "gif", "bmp": "bmp", "webp": "webp"}
        mime = mime_map.get(ext, "jpeg")
        with open(path, "rb") as f:
            data = f.read()
        return f"data:image/{mime};base64,{base64.b64encode(data).decode('utf-8')}"
    except Exception:
        return None


def allowed_file(filename, extensions=None):
    """Check if file extension is allowed."""
    if extensions is None:
        extensions = {"csv", "xlsx", "xls", "json", "txt", "md"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def save_uploaded_file(file, upload_folder):
    """Save an uploaded file to disk and return its path."""
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    safe_name = f"{name}_{ts}{ext}"
    path = os.path.join(upload_folder, safe_name)
    file.save(path)
    return path


def format_response(success=True, data=None, message=None, error=None):
    """Build a uniform JSON API response."""
    resp = {"success": success, "timestamp": datetime.now().isoformat()}
    if data is not None:
        resp["data"] = data
    if message:
        resp["message"] = message
    if error:
        resp["error"] = error
    return resp


def truncate_text(text, max_length=500):
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def generate_session_id():
    """Generate a short session identifier."""
    raw = f"{datetime.now().timestamp()}{os.urandom(4)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
