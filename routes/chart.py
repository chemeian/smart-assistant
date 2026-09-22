"""Chart API route -- generates visual charts for uploaded data files."""
import os
from flask import Blueprint, request, jsonify

from services.chart_service import ChartService
from utils.helpers import format_response

chart_bp = Blueprint("chart", __name__, url_prefix="/api/chart")
_service = ChartService()


@chart_bp.route("/generate", methods=["POST"])
def generate_charts():
    """Generate chart images for a previously uploaded dataset."""
    data = request.get_json(silent=True) or {}
    file_path = data.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        return jsonify(format_response(success=False, error="文件不存在")), 400
    result = _service.generate_charts(file_path)
    if not result["success"]:
        return jsonify(format_response(success=False, error=result.get("error", "生成失败"))), 400
    return jsonify(format_response(success=True, data=result["data"]))


@chart_bp.route("/stats", methods=["POST"])
def get_stats():
    """Return basic statistics for a dataset."""
    data = request.get_json(silent=True) or {}
    file_path = data.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        return jsonify(format_response(success=False, error="文件不存在")), 400
    result = _service.generate_charts(file_path)
    if not result["success"]:
        return jsonify(format_response(success=False, error=result.get("error", "获取失败"))), 400
    return jsonify(format_response(success=True, data={
        "stats": result["data"]["stats"],
        "shape": result["data"]["shape"],
        "numeric_cols": result["data"]["numeric_cols"],
        "text_cols": result["data"]["text_cols"],
    }))
