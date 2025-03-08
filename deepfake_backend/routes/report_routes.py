from flask import Blueprint, jsonify, request
from database.db import mongo
from utils.pdf_generator import generate_pdf_report
import datetime

report_bp = Blueprint('report', __name__)

@report_bp.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    if not data or "file_id" not in data or "probability" not in data:
        return jsonify({"error": "Missing required data"}), 400

    file_id = data["file_id"]
    result_data = {
        "probability": data["probability"],
        "confidence_score": data["confidence_score"],
        "explanation": data["explanation"]
    }

    # Generate and save the report
    pdf_path = generate_pdf_report(result_data, file_id)

    # Save report metadata in MongoDB
    report_data = {
        "file_id": file_id,
        "pdf_path": pdf_path,
        "generated_at": datetime.datetime.utcnow(),
        "probability": result_data["probability"],
        "confidence_score": result_data["confidence_score"],
        "explanation": result_data["explanation"]
    }
    mongo.db.reports.insert_one(report_data)

    return jsonify({"message": "Report generated successfully", "file_id": file_id, "pdf_path": pdf_path}), 201
