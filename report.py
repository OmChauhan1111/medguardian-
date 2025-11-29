# report.py (UPDATED)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os

# -------------------- Normal ranges (readable) --------------------
normal_ranges = {
    # HEART
    "Chest Pain Type": "0 - 3",
    "Resting Blood Pressure": "90 - 120",
    "Cholesterol": "< 200",
    "Fasting Blood Sugar": "< 100",
    "Resting ECG": "0 - 2",
    "Max Heart Rate": "60 - 100",
    "Exercise Induced Angina": "No",
    "ST Depression": "0 - 1",
    "ST Slope": "0 - 2",
    "Major Vessels Colored": "0 - 1",
    "Thalassemia": "0 - 1",

    # DIABETES
    "BMI": "18.5 - 24.9",
    "Blood Glucose": "70 - 99",
    "HbA1c": "4 - 5.6",
    "Hypertension": "No",

    # KIDNEY
    "Blood Pressure": "90 - 120",
    "Specific Gravity": "1.005 - 1.030",
    "Albumin": "0 - 1",
    "Sugar": "0 - 1",
    "Red Blood Cells": "Normal",
    "Pus Cell": "Normal",
    "Pus Cell Clumps": "No",
    "Bacteria": "No",
    "Blood Glucose Random": "70 - 140",
    "Blood Urea": "7 - 20",
    "Serum Creatinine": "0.6 - 1.2",
    "Sodium": "135 - 145",
    "Potassium": "3.5 - 5.0",
    "Haemoglobin": "12 - 17",
    "Packed Cell Volume": "38 - 52",
    "White Blood Cell Count": "4000 - 11000",
    "Red Blood Cell Count": "4.5 - 6.0",
    "Diabetes Mellitus": "-",
    "Coronary Artery Disease": "-",
    "Appetite": "-",
    "Peda Edema": "-",
    "Anaemia": "-"
}

# -------------------- Key rename mapping --------------------
rename_keys = {
    # HEART
    "cp": "Chest Pain Type",
    "restbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels Colored",
    "thal": "Thalassemia",

    # DIABETES
    "bmi": "BMI",
    "glucose": "Blood Glucose",
    "hba": "HbA1c",
    "hyt": "Hypertension",

    # KIDNEY
    "bp": "Blood Pressure",
    "sg": "Specific Gravity",
    "al": "Albumin",
    "su": "Sugar",
    "red_blood_cells": "Red Blood Cells",
    "pc": "Pus Cell",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "bgr": "Blood Glucose Random",
    "bu": "Blood Urea",
    "sc": "Serum Creatinine",
    "sod": "Sodium",
    "pot": "Potassium",
    "hemo": "Haemoglobin",
    "pcv": "Packed Cell Volume",
    "wc": "White Blood Cell Count",
    "rc": "Red Blood Cell Count",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Peda Edema",
    "ane": "Anaemia"
}

# -------------------- Safe flag checker --------------------
def check_flag(param, value):
    """
    Return "Normal"/"High"/"Low" based on normal_ranges and simple heuristics.
    """
    if value is None:
        return "Normal"
    s = str(value).strip()
    if s == "":
        return "Normal"
    low_s = s.lower()

    # simple string heuristics
    if low_s in ["yes", "poor", "abnormal", "high", "true", "1"]:
        return "High"
    if low_s in ["no", "good", "normal", "false", "0"]:
        return "Normal"

    # numeric checks when normal_ranges available
    if param in normal_ranges:
        rg = normal_ranges[param]
        try:
            if "<" in rg:
                limit = float(rg.replace("<", "").strip())
                try:
                    return "High" if float(s) > limit else "Normal"
                except:
                    return "Normal"
            if "-" in rg:
                parts = rg.split("-")
                if len(parts) == 2:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    try:
                        val = float(s)
                        if val < low:
                            return "Low"
                        if val > high:
                            return "High"
                        return "Normal"
                    except:
                        return "Normal"
        except Exception:
            return "Normal"
    return "Normal"

# -------------------- Watermark --------------------
def draw_watermark(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 40)
    # light gray with transparency (reportlab doesn't support alpha in setFillColorRGB for all backends,
    # keep light color)
    canvas.setFillColorRGB(0.88, 0.88, 0.88)
    canvas.translate(300, 400)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "MedGuardian")
    canvas.restoreState()

# -------------------- Utility: safe string formatting --------------------
def s(val):
    """Safe string conversion for table cells."""
    if val is None:
        return "-"
    if isinstance(val, float):
        # trim excessive decimal
        if val.is_integer():
            return str(int(val))
    return str(val)

# -------------------- Main: generate_pdf_report --------------------
def generate_pdf_report(disease, data, diagnosis, filename="Medical_Report.pdf"):
    """
    disease: string, e.g. "Heart"
    data: dict containing patient fields and raw values (may have keys like 'Phone' or 'Patient Contact')
    diagnosis: short string summary to display
    filename: output filename (relative)
    Returns: filename
    """
    # defensive copy and rename keys to readable names
    fixed = {}
    for k, v in (data or {}).items():
        # map keys if present in rename_keys (both original keys and already readable kept)
        new_k = rename_keys.get(k, k)
        fixed[new_k] = v

    # Ensure common convenience fields exist (normalize phone keys)
    contact_val = (
        fixed.get("Phone")
        or fixed.get("Patient Contact")
        or fixed.get("Contact")
        or fixed.get("Phone Number")
        or fixed.get("Patient Phone")
        or "-"
    )

    # also safe patient name keys
    patient_id = fixed.get("Patient ID") or fixed.get("patient_id") or "-"
    patient_name = fixed.get("Patient Name") or fixed.get("patient_name") or "-"
    doctor_name = fixed.get("Doctor Name") or fixed.get("doctor_name") or "-"
    referred_by = fixed.get("Referred By") or fixed.get("referred_by") or "-"

    # Build document
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    # Header: left text + logo (if available)
    header_left_html = (
        "<b><font size=16 color='darkblue'>MedGuardian</font></b><br/>"
        f"<font size=8><b>Contact: {s(contact_val)} • Chennai, Tamil Nadu</b></font><br/>"
        "<font size=8><b>Email: info@medguardian.com</b></font>"
    )
    # Try to load image; fallback to text if image missing
    logo_path = "logo.png"
    logo_obj = None
    if os.path.exists(logo_path):
        try:
            logo_obj = Image(logo_path, width=110, height=55)
        except Exception:
            logo_obj = None

    header_table = [[Paragraph(header_left_html, styles["Normal"]), logo_obj or Paragraph("", styles["Normal"])]]
    head = Table(header_table, colWidths=[380, 100])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    elements.append(head)
    elements.append(Spacer(1, 8))

    # Title
    elements.append(Paragraph("<b>PATIENT MEDICAL REPORT</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    # Patient info table
    patient_info = [
        ["Patient ID", s(patient_id)],
        ["Patient Name", s(patient_name)],
        ["Contact", s(contact_val)],
        ["Age", s(fixed.get("Age") or "-")],
        ["Gender", s(fixed.get("Gender") or "-")],
        ["Test Type", s(disease or "-")],
        ["Doctor Name", s(doctor_name)],
        ["Referred By", s(referred_by)],
        ["Report Date", datetime.now().strftime("%d-%m-%Y %I:%M %p")],
    ]
    elements.append(Table(patient_info, colWidths=[160, 320], hAlign="LEFT"))
    elements.append(Spacer(1, 12))

    # Results
    elements.append(Paragraph("<b>Test Results</b>", styles["Heading3"]))
    # Build result rows by checking fixed dict keys present in normal_ranges
    result_table = [["Parameter", "Value", "Normal Range", "Status"]]
    for key, val in fixed.items():
        # Only include keys that are in our normal_ranges mapping (readable names)
        if key in normal_ranges:
            status = check_flag(key, val)
            result_table.append([key, s(val), normal_ranges.get(key, "-"), status])

    # If none found, show a friendly message
    if len(result_table) == 1:
        elements.append(Paragraph("No numeric test parameters were provided for range checking.", styles["Normal"]))
    else:
        t = Table(result_table, colWidths=[170, 80, 160, 60], hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8ff")),
                    ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        elements.append(t)
    elements.append(Spacer(1, 10))

    # AI diagnosis summary box
    elements.append(Paragraph("<b>AI Diagnosis Summary</b>", styles["Heading3"]))
    box = Table([[Paragraph(s(diagnosis), ParagraphStyle(name="diag", fontSize=10, alignment=0))]], colWidths=[480])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.blue), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef7ff"))]))
    elements.append(box)
    elements.append(Spacer(1, 18))

    # Signature area
    sig_style = ParagraphStyle(name="sig", alignment=2, fontSize=10)
    elements.append(Paragraph("<b>Om Singh Chauhan</b>", sig_style))
    elements.append(Paragraph("________________", sig_style))
    elements.append(Paragraph("<font size=8>(AI Generated Signature)</font>", sig_style))

    elements.append(Spacer(1, 18))
    elements.append(HRFlowable(width="100%", thickness=0.6))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Note: This report is AI generated. Consult a doctor for clinical confirmation.", ParagraphStyle(name="note", alignment=1, fontSize=8)))

    # Build PDF with watermark
    doc.build(elements, onFirstPage=draw_watermark)
    return filename

# If run directly, quick test (creates sample PDF)
if __name__ == "__main__":
    sample_data = {
        "Patient ID": "MG-TEST-0001",
        "Patient Name": "Test Patient",
        "Phone": "+91 9876543210",
        "Age": 35,
        "Gender": "Male",
        "Chest Pain Type": 1,
        "Resting Blood Pressure": 130,
        "Cholesterol": 210,
        "HbA1c": 6.2,
        "Blood Glucose Random": 150
    }
    print("Generating sample report 'sample_report.pdf' ...")
    generate_pdf_report("Heart", sample_data, "Sample diagnosis: Minor risk detected", filename="sample_report.pdf")
    print("Done.")
