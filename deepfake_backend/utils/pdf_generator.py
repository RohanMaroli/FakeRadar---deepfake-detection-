from fpdf import FPDF
import datetime

def generate_pdf_report(result_data, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Deepfake Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Probability of Deepfake: {result_data['probability']}%", ln=True)
    pdf.cell(0, 8, f"Confidence Score: {result_data['confidence_score']}%", ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, f"Explanation: {result_data['explanation']}", ln=True)
    pdf.ln(10)

    pdf.cell(0, 8, f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    pdf.output(output_path)
