"""
NLP API route - text analysis and processing endpoints.
"""
from flask import Blueprint, request, jsonify

from services.nlp_service import NLPService
from utils.helpers import format_response

nlp_bp = Blueprint("nlp", __name__, url_prefix="/api/nlp")
_service = NLPService()


@nlp_bp.route("/sentiment", methods=["POST"])
def sentiment():
    """Analyse the sentiment of a given text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = _service.sentiment_analysis(text)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/keywords", methods=["POST"])
def keywords():
    """Extract keywords from text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    top_k = int(data.get("top_k", 10))
    result = _service.keyword_extraction(text, top_k)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/summary", methods=["POST"])
def summary():
    """Generate an extractive text summary."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    max_sentences = int(data.get("max_sentences", 3))
    result = _service.text_summary(text, max_sentences)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/word-count", methods=["POST"])
def word_count():
    """Count characters, words, sentences, and readability stats."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = _service.word_count(text)
    return jsonify(result)


@nlp_bp.route("/classify", methods=["POST"])
def classify_text():
    """Classify text into category labels."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = _service.classify_text(text)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/entities", methods=["POST"])
def extract_entities():
    """Extract named entities (URLs, emails, phones, dates, etc.) from text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = _service.extract_entities(text)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/readability", methods=["POST"])
def readability():
    """Calculate readability score and reading time estimate."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = _service.readability_score(text)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@nlp_bp.route("/analyze", methods=["POST"])
def full_analyze():
    """Run comprehensive NLP analysis (sentiment, keywords, summary, classify, entities, readability)."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify(format_response(success=False, error="Text cannot be empty")), 400

    return jsonify(format_response(success=True, data={
        "sentiment": _service.sentiment_analysis(text).get("data"),
        "keywords": _service.keyword_extraction(text).get("data"),
        "summary": _service.text_summary(text).get("data"),
        "word_count": _service.word_count(text).get("data"),
        "classification": _service.classify_text(text).get("data"),
        "entities": _service.extract_entities(text).get("data"),
        "readability": _service.readability_score(text).get("data"),
        "text_length": len(text),
    }))