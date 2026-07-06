import os
import sys
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import shutil
from pypdf import PdfReader

# Ensure 'src' is in path so we can import the crew package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from medical_decision_support_assistant.crew import MedicalDecisionSupportAssistantCrew
# pyrefly: ignore [missing-import]
from database import init_db, get_db, Patient, Report, AnalysisHistory, SessionLocal
from rag import retrieve_guidelines, build_faiss_index
from pdf_generator import generate_pdf_report, REPORTS_DIR
from crewai.tools import tool
from crewai import LLM

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Database tables
init_db()

# Build FAISS RAG index on startup
try:
    if not os.path.exists("./data/faiss.index"):   # <-- change filename if yours differs
        build_faiss_index()
except Exception as e:
    print(f"Warning: Could not build FAISS index on startup: {e}")

app = FastAPI(title="Medical Decision Support Assistant API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Search tool wrapper for local FAISS guidelines
@tool("SearchMedicalGuidelines")
def search_medical_guidelines(query: str) -> str:
    """Searches the local FAISS medical guidelines index for relevant protocols, guidelines, and diagnostic workups."""
    results = retrieve_guidelines(query, k=4)
    if not results:
        return "No relevant local guidelines found."
    return "\n\n".join([r["content"] for r in results])


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class VitalsSchema(BaseModel):
    temperature: Optional[float] = None
    pulse: Optional[int] = None
    blood_pressure: Optional[str] = None
    oxygen_saturation: Optional[float] = None


class PatientAnalysisInput(BaseModel):
    patient_name: str
    age: int = Field(gt=0, le=120)
    gender: str
    weight: Optional[float] = None
    height: Optional[float] = None
    vitals: Optional[VitalsSchema] = None
    symptoms: str
    current_medications: Optional[str] = ""
    allergies: Optional[str] = ""
    known_diseases: Optional[str] = ""
    family_history: Optional[str] = ""
    medical_history: Optional[str] = ""
    doctor_notes: Optional[str] = ""


# ─── Agent Execution Order ───────────────────────────────────────────────────

AGENT_ORDER = [
    "senior_medical_history_analyst",
    "clinical_symptom_specialist",
    "medical_literature_and_guidelines_specialist",
    "clinical_pharmacologist_and_drug_safety_specialist",
    "clinical_risk_stratification_expert",
    "senior_clinical_documentation_specialist",
]

TASKS_TO_AGENT = {
    "analyze_patient_profile": "senior_medical_history_analyst",
    "interpret_symptoms_and_generate_differential_diagnosis": "clinical_symptom_specialist",
    "retrieve_relevant_medical_guidelines_and_protocols": "medical_literature_and_guidelines_specialist",
    "assess_drug_interactions_and_allergy_risks": "clinical_pharmacologist_and_drug_safety_specialist",
    "assess_patient_risk_level": "clinical_risk_stratification_expert",
    "generate_clinical_decision_support_report": "senior_clinical_documentation_specialist",
}


def _get_active_llm() -> LLM:
    """Return the best available LLM based on configured API keys."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        return LLM(model="gemini/gemini-2.0-flash", api_key=gemini_key)
    elif groq_key:
        return LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_key)
    elif openai_key:
        return LLM(model="openai/gpt-4o-mini", api_key=openai_key)
    else:
        raise ValueError(
            "No API key found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in .env"
        )


def update_agent_status(analysis_id: int, agent_name: str, status: str, db: Session):
    record = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
    if record:
        states = record.get_agent_states()
        states[agent_name] = status

        # Progress the next pending agent to "running"
        if status == "completed":
            try:
                idx = AGENT_ORDER.index(agent_name)
                if idx + 1 < len(AGENT_ORDER):
                    states[AGENT_ORDER[idx + 1]] = "running"
            except ValueError:
                pass

        record.set_agent_states(states)
        db.commit()


def run_crew_orchestration(analysis_id: int, inputs: dict, db_session_factory):
    db = db_session_factory()
    try:
        # Resolve LLM
        try:
            active_llm = _get_active_llm()
        except ValueError as key_err:
            raise RuntimeError(str(key_err))

        # Instantiate the crew
        crew_instance = MedicalDecisionSupportAssistantCrew()

        # Inject local RAG search tool into literature specialist
        try:
            lit_agent = crew_instance.medical_literature_and_guidelines_specialist()
            if search_medical_guidelines not in lit_agent.tools:
                lit_agent.tools.append(search_medical_guidelines)
        except Exception as e:
            print(f"Could not inject RAG search tool: {e}")

        crew = crew_instance.crew()

        # Override every agent's LLM to the active one
        for agent in crew.agents:
            agent.llm = active_llm
        if hasattr(crew, "chat_llm") and crew.chat_llm:
            crew.chat_llm = active_llm

        # Attach task callbacks to update live agent timeline
        for task in crew.tasks:
            matched_agent = None
            for key, agent_name in TASKS_TO_AGENT.items():
                if (
                    key in str(task.description).lower()
                    or key in str(task.expected_output).lower()
                    or (task.agent and agent_name in str(task.agent.role).lower().replace(" ", "_"))
                ):
                    matched_agent = agent_name
                    break

            if not matched_agent:
                # Fallback: use positional index
                idx = crew.tasks.index(task)
                if idx < len(AGENT_ORDER):
                    matched_agent = AGENT_ORDER[idx]

            if matched_agent:
                def _make_callback(an=matched_agent):
                    def _cb(output):
                        _db = db_session_factory()
                        try:
                            update_agent_status(analysis_id, an, "completed", _db)
                        finally:
                            _db.close()
                    return _cb
                task.callback = _make_callback()

        # Mark first agent as running
        update_agent_status(analysis_id, AGENT_ORDER[0], "running", db)

        # Execute the crew
        result = crew.kickoff(inputs=inputs)

        # Persist results
        record = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
        if record:
            record.status = "completed"
            record.completed_at = datetime.utcnow()
            record.result_markdown = result.raw
            record.set_agent_states({agent: "completed" for agent in AGENT_ORDER})

            try:
                pdf_filename = f"report_analysis_{analysis_id}_{int(datetime.utcnow().timestamp())}.pdf"
                generate_pdf_report(result.raw, pdf_filename)

                raw_lower = result.raw.lower()
                if "critical" in raw_lower:
                    risk_level = "Critical"
                elif "high risk" in raw_lower or "high-risk" in raw_lower:
                    risk_level = "High"
                elif "moderate" in raw_lower or "medium" in raw_lower:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"

                record.result_json = json.dumps({
                    "risk_level": risk_level,
                    "pdf_report_path": f"/api/reports/{pdf_filename}",
                })
            except Exception as pdf_err:
                print(f"Error generating PDF report: {pdf_err}")

            db.commit()

    except Exception as exc:
        print(f"Crew execution failed for analysis {analysis_id}: {exc}")
        record = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
        if record:
            record.status = "failed"
            record.completed_at = datetime.utcnow()
            record.result_markdown = f"Analysis failed: {str(exc)}"
            states = record.get_agent_states()
            for agent, status in states.items():
                if status == "running":
                    states[agent] = "failed"
            record.set_agent_states(states)
            db.commit()
    finally:
        db.close()


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    any_key = bool(openai_key or groq_key or gemini_key)
    return {
        "status": "healthy",
        "api_keys_configured": {
            "openai": bool(openai_key),
            "groq": bool(groq_key),
            "gemini": bool(gemini_key),
            "any": any_key,
        },
        "database_url": os.getenv("DATABASE_URL", "sqlite:///./medical_assistant.db"),
    }


@app.post("/analyze")
def analyze(
    input_data: PatientAnalysisInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Verify at least one API key is present
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not openai_key and not groq_key and not gemini_key:
        raise HTTPException(
            status_code=400,
            detail="Missing API credentials. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in .env",
        )

    # Upsert patient
    patient = db.query(Patient).filter(Patient.name == input_data.patient_name).first()
    vitals = input_data.vitals or VitalsSchema()

    if not patient:
        patient = Patient(
            name=input_data.patient_name,
            age=input_data.age,
            gender=input_data.gender,
            weight=input_data.weight,
            height=input_data.height,
            temperature=vitals.temperature,
            pulse=vitals.pulse,
            blood_pressure=vitals.blood_pressure,
            oxygen_saturation=vitals.oxygen_saturation,
            medical_history=input_data.medical_history,
            current_medications=input_data.current_medications,
            allergies=input_data.allergies,
            known_diseases=input_data.known_diseases,
            family_history=input_data.family_history,
            doctor_notes=input_data.doctor_notes,
        )
        db.add(patient)
    else:
        patient.age = input_data.age
        patient.gender = input_data.gender
        patient.weight = input_data.weight or patient.weight
        patient.height = input_data.height or patient.height
        patient.temperature = vitals.temperature or patient.temperature
        patient.pulse = vitals.pulse or patient.pulse
        patient.blood_pressure = vitals.blood_pressure or patient.blood_pressure
        patient.oxygen_saturation = vitals.oxygen_saturation or patient.oxygen_saturation
        patient.medical_history = input_data.medical_history or patient.medical_history
        patient.current_medications = input_data.current_medications
        patient.allergies = input_data.allergies
        patient.known_diseases = input_data.known_diseases
        patient.family_history = input_data.family_history
        patient.doctor_notes = input_data.doctor_notes

    db.commit()
    db.refresh(patient)

    # Create analysis record
    initial_states = {agent: "pending" for agent in AGENT_ORDER}
    analysis = AnalysisHistory(
        patient_id=patient.id,
        status="running",
        agent_states=json.dumps(initial_states),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Build crew inputs
    crew_inputs = {
        "patient_name": patient.name,
        "age": str(patient.age),
        "gender": patient.gender,
        "temperature": str(patient.temperature or "Not recorded"),
        "pulse": str(patient.pulse or "Not recorded"),
        "blood_pressure": patient.blood_pressure or "Not recorded",
        "oxygen_saturation": str(patient.oxygen_saturation or "Not recorded"),
        "symptoms": input_data.symptoms,
        "current_medications": patient.current_medications or "None",
        "allergies": patient.allergies or "None",
        "known_diseases": patient.known_diseases or "None",
        "family_history": patient.family_history or "None",
        "weight": str(patient.weight or "Not recorded"),
        "height": str(patient.height or "Not recorded"),
        "medical_history": patient.medical_history or "None",
        "doctor_notes": patient.doctor_notes or "None",
    }

    background_tasks.add_task(
        run_crew_orchestration,
        analysis.id,
        crew_inputs,
        SessionLocal,
    )

    return {
        "analysis_id": analysis.id,
        "patient_id": patient.id,
        "status": "running",
        "agent_states": initial_states,
    }


@app.post("/upload-report")
def upload_report(
    patient_id: Optional[int] = Form(None),
    patient_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    os.makedirs("./data/uploads", exist_ok=True)
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    file_path = f"./data/uploads/{datetime.utcnow().timestamp()}_{safe_name}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    content_summary = ""
    try:
        reader = PdfReader(file_path)
        extracted_text = ""
        for page in reader.pages[:3]:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        content_summary = extracted_text[:2000]
    except Exception as e:
        content_summary = f"Could not parse PDF text: {str(e)}"

    db_patient_id = patient_id
    if not db_patient_id and patient_name:
        patient = db.query(Patient).filter(Patient.name == patient_name).first()
        if patient:
            db_patient_id = patient.id

    report = Report(
        patient_id=db_patient_id,
        title=file.filename,
        file_path=file_path,
        content_summary=content_summary,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Re-index into FAISS
    try:
        rag_copy = os.path.join("./data", f"upload_{report.id}_{safe_name}")
        shutil.copyfile(file_path, rag_copy)
        build_faiss_index()
    except Exception as e:
        print(f"Warning: Could not re-index uploaded PDF in FAISS: {e}")

    return {
        "report_id": report.id,
        "title": report.title,
        "content_summary": content_summary,
        "indexed": True,
    }


@app.get("/history")
def history(db: Session = Depends(get_db)):
    results = db.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc()).all()
    history_list = []
    for item in results:
        patient = item.patient
        result_details: dict = {}
        if item.result_json:
            try:
                result_details = json.loads(item.result_json)
            except Exception:
                pass
        history_list.append({
            "id": item.id,
            "patient_name": patient.name if patient else "Unknown",
            "age": patient.age if patient else 0,
            "gender": patient.gender if patient else "Unknown",
            "status": item.status,
            "risk_level": result_details.get("risk_level", "Unknown"),
            "pdf_report_path": result_details.get("pdf_report_path"),
            "created_at": item.created_at.isoformat(),
            "completed_at": item.completed_at.isoformat() if item.completed_at else None,
        })
    return history_list


@app.get("/history/{id}")
def history_detail(id: int, db: Session = Depends(get_db)):
    item = db.query(AnalysisHistory).filter(AnalysisHistory.id == id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Analysis record not found")

    patient = item.patient
    result_details: dict = {}
    if item.result_json:
        try:
            result_details = json.loads(item.result_json)
        except Exception:
            pass

    return {
        "id": item.id,
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "weight": patient.weight,
            "height": patient.height,
            "temperature": patient.temperature,
            "pulse": patient.pulse,
            "blood_pressure": patient.blood_pressure,
            "oxygen_saturation": patient.oxygen_saturation,
            "medical_history": patient.medical_history or "",
            "current_medications": patient.current_medications or "",
            "allergies": patient.allergies or "",
            "known_diseases": patient.known_diseases or "",
            "family_history": patient.family_history or "",
            "doctor_notes": patient.doctor_notes or "",
        } if patient else None,
        "status": item.status,
        "agent_states": item.get_agent_states(),
        "result_markdown": item.result_markdown,
        "risk_level": result_details.get("risk_level", "Unknown"),
        "pdf_report_path": result_details.get("pdf_report_path"),
        "created_at": item.created_at.isoformat(),
        "completed_at": item.completed_at.isoformat() if item.completed_at else None,
    }


@app.get("/reports/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF report not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)