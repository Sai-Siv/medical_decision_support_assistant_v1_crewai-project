import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DATA_DIR = "./data"
INDEX_PATH = "./data/faiss_index.index"
CHUNKS_PATH = "./data/chunks.txt"

# Initialize SentenceTransformer model
print("Loading sentence-transformers/all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_sample_guideline_pdf(filename, title, sections):
    """Creates a professional sample medical guideline PDF using ReportLab."""
    filepath = os.path.join(DATA_DIR, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor="#1e3a8a",
        spaceAfter=15,
    )
    heading_style = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor="#0f766e",
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor="#1f2937",
        spaceAfter=8,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))

    for sec_title, sec_content in sections:
        story.append(Paragraph(sec_title, heading_style))
        for paragraph in sec_content:
            story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Created sample PDF: {filepath}")


def check_and_generate_sample_data():
    """Generates sample medical guidelines if none exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if existing_files:
        print(f"Found existing guideline PDFs: {existing_files}")
        return

    print("No guideline PDFs found. Creating sample medical guideline PDFs...")

    # 1. Hypertension guidelines
    create_sample_guideline_pdf(
        "hypertension_guidelines.pdf",
        "Clinical Guidelines for the Management of Hypertension (JNC-8/AHA)",
        [
            (
                "1. Classification of Blood Pressure",
                [
                    "Normal Blood Pressure: Systolic less than 120 mmHg and diastolic less than 80 mmHg.",
                    "Elevated Blood Pressure: Systolic 120-129 mmHg and diastolic less than 80 mmHg.",
                    "Stage 1 Hypertension: Systolic 130-139 mmHg or diastolic 80-89 mmHg.",
                    "Stage 2 Hypertension: Systolic 140 mmHg or higher, or diastolic 90 mmHg or higher.",
                    "Hypertensive Crisis: Systolic over 180 mmHg and/or diastolic over 120 mmHg, requiring immediate medical attention.",
                ],
            ),
            (
                "2. Treatment Targets and Thresholds",
                [
                    "For adults with confirmed hypertension and known cardiovascular disease or 10-year ASCVD risk of 10% or higher, the target BP is less than 130/80 mmHg.",
                    "For patients with diabetes mellitus or chronic kidney disease (CKD), the target blood pressure is also less than 130/80 mmHg.",
                    "Pharmacological therapy should be initiated at blood pressure readings >= 140/90 mmHg for low-risk individuals, and >= 130/80 mmHg for high-risk individuals.",
                ],
            ),
            (
                "3. Pharmacological First-Line Therapies",
                [
                    "First-line agents include: Thiazide diuretics (e.g., Chlorthalidone, Hydrochlorothiazide), Calcium Channel Blockers (CCBs like Amlodipine, Nifedipine), and Angiotensin-Converting Enzyme (ACE) Inhibitors (e.g., Lisinopril, Enalapril) or Angiotensin Receptor Blockers (ARBs like Losartan, Valsartan).",
                    "Do not combine ACE inhibitors and ARBs due to increased risk of hyperkalemia, renal impairment, and acute kidney injury.",
                    "For patients of African descent without CKD, initial therapy should include a thiazide diuretic or calcium channel blocker.",
                ],
            ),
        ],
    )

    # 2. Asthma Guidelines
    create_sample_guideline_pdf(
        "asthma_guidelines.pdf",
        "Global Initiative for Asthma (GINA) Clinical Management Guidelines",
        [
            (
                "1. Diagnosis and Assessment",
                [
                    "Asthma is characterized by chronic airway inflammation. Symptoms include wheezing, shortness of breath, chest tightness, and cough that vary over time and in intensity.",
                    "Objective verification of airflow limitation is required (Spirometry showing FEV1/FVC ratio < 0.75-0.80).",
                    "Severity is classified based on the level of treatment required to control symptoms: Mild (Steps 1-2), Moderate (Step 3-4), or Severe (Step 5).",
                ],
            ),
            (
                "2. Stepwise Pharmacological Management",
                [
                    "Step 1-2: Preferred controller and reliever is low-dose ICS-formoterol (Inhaled Corticosteroid / Formoterol) taken as needed for symptom relief.",
                    "Step 3: Low-dose maintenance ICS-formoterol (maintenance and reliever therapy, SMART). Alternatively, low-dose ICS-LABA daily controller plus SABA as needed.",
                    "Step 4: Medium-dose maintenance ICS-formoterol SMART therapy or medium-dose ICS-LABA daily controller plus SABA.",
                    "Step 5: High-dose ICS-LABA, refer for phenotypic evaluation and add-on treatments (e.g., LAMA like Tiotropium, biologics targeting IgE, IL-5, or IL-4R).",
                ],
            ),
            (
                "3. Emergency Protocol and Red Flags",
                [
                    "Severe exacerbations are characterized by: speech limited to words, patient sitting hunched forward, respiratory rate > 30/min, accessory muscle use, heart rate > 120/min, or oxygen saturation < 90% on room air.",
                    "Immediate management: High-dose inhaled SABA (albuterol) and ipratropium via MDI + spacer, systemic corticosteroids (prednisolone 1mg/kg up to 50mg), and oxygen therapy to maintain SpO2 93-95% (94-98% in pregnant females).",
                ],
            ),
        ],
    )

    # 3. Diabetes Guidelines
    create_sample_guideline_pdf(
        "diabetes_guidelines.pdf",
        "American Diabetes Association (ADA) Standards of Care in Diabetes",
        [
            (
                "1. Diagnostic Standards",
                [
                    "Diabetes is diagnosed if Fasting Plasma Glucose (FPG) is >= 126 mg/dL, 2-hour Post-prandial Glucose is >= 200 mg/dL during an OGTT, or HbA1c is >= 6.5%.",
                    "Prediabetes: HbA1c 5.7% to 6.4%, or Fasting Glucose 100 to 125 mg/dL.",
                ],
            ),
            (
                "2. Glycemic Goals and Monitoring",
                [
                    "For non-pregnant adults, a reasonable glycemic goal is HbA1c < 7.0% (53 mmol/mol).",
                    "Pre-prandial capillary plasma glucose target: 80-130 mg/dL.",
                    "Peak post-prandial capillary plasma glucose target: < 180 mg/dL.",
                ],
            ),
            (
                "3. Pharmacological Management Strategies",
                [
                    "Metformin is the preferred initial pharmacological agent for the treatment of type 2 diabetes, provided there are no contraindications (e.g., eGFR < 30 mL/min/1.73m2).",
                    "Early combination therapy should be considered in patients with HbA1c >= 1.5% to 2.0% above their glycemic target.",
                    "In patients with type 2 diabetes and established atherosclerotic cardiovascular disease (ASCVD), high cardiovascular risk, heart failure, or CKD, the treatment regimen should include a GLP-1 receptor agonist or an SGLT2 inhibitor.",
                ],
            ),
        ],
    )


def build_faiss_index():
    """Reads PDF files, extracts text, chunks them, computes embeddings, and builds FAISS index."""
    check_and_generate_sample_data()

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    all_chunks = []

    for filename in pdf_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            reader = PdfReader(filepath)
            print(f"Parsing: {filepath}")
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                # Basic chunking: split by sentences/lines into chunks of ~500 chars
                lines = text.split("\n")
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) < 500:
                        current_chunk += line + " "
                    else:
                        if current_chunk.strip():
                            all_chunks.append(
                                f"Source: [{filename} (Page {page_num + 1})] - {current_chunk.strip()}"
                            )
                        current_chunk = line + " "
                if current_chunk.strip():
                    all_chunks.append(
                        f"Source: [{filename} (Page {page_num + 1})] - {current_chunk.strip()}"
                    )
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not all_chunks:
        print("Warning: No medical guideline text extracted.")
        return

    # Write chunks for later loading
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.replace("\n", " ") + "\n")

    print(f"Extracted {len(all_chunks)} guideline chunks. Embedding and creating FAISS index...")

    # Embed chunks
    embeddings = embedding_model.encode(all_chunks)
    embeddings = np.array(embeddings, dtype="float32")

    # Build Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print("FAISS index saved successfully.")


def retrieve_guidelines(query: str, k: int = 3):
    """Queries the FAISS index to find relevant guideline passages."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("FAISS index or chunks not found. Building now...")
        build_faiss_index()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return []

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = [line.strip() for line in f.readlines()]

        # Embed query
        query_vector = embedding_model.encode([query])
        query_vector = np.array(query_vector, dtype="float32")

        # Search
        distances, indices = index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(chunks):
                results.append(
                    {"content": chunks[idx], "distance": float(distances[0][i])}
                )
        return results
    except Exception as e:
        print(f"Error retrieving from RAG index: {e}")
        return []


# Build index automatically on initial run if not present
if __name__ == "__main__":
    build_faiss_index()
