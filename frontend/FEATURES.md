# 🌟 EUNOCURA Features Overview

> **Read this first.** Sections say plainly whether they are **Implemented** (backend +
> frontend, demoable) or **Fixture** (a real screen rendering `src/data/dummyData.js`,
> labelled with an amber banner in the UI). Anything not marked Implemented is not
> wired to a backend route, and no section below claims otherwise.

## 👨‍⚕️ Doctor Portal Features

### 1. Dashboard
**What it does:**
- Displays daily statistics (patients, reports, appointments, alerts)
- Shows current patient information
- Provides quick action buttons
- Lists recent activity

**User Value:**
- At-a-glance overview of workload
- Quick access to common tasks
- Organized workflow

---

### 2. Audio Recording & Transcription
**What it does:**
- Records doctor-patient conversations
- Auto-transcribes audio to text using AI
- Allows doctors to add written notes
- Timestamps all recordings

**User Value:**
- Eliminates manual note-taking during consultation
- More time for patient interaction
- Accurate conversation records
- Reduces documentation burden

**Technical Implementation (Implemented):**
- Local `faster-whisper` on CPU by default; no cloud speech provider at all
- Text-only degrade when neither is available — the consultation screen keeps working
- WebSocket streaming endpoint for live dictation; REST endpoint for uploads

---

### 3. Clinical Report Generation
**What it does:**
- AI generates structured clinical reports
- Combines transcription + doctor's notes + patient history
- Editable reports with preview mode
- Export to PDF, print, email

**User Value:**
- Saves 30+ minutes per patient
- Consistent, professional formatting
- Reduces documentation errors
- ICD-10 coding suggestions

**Technical Implementation (Partially implemented):**
- Structurally generated from the transcript + notes; export is client-side
- ICD-10 coding suggestions and version history are **not** implemented

---

### 4. Test Results & Trend Analysis
**What it does:**
- Upload test results (PDFs, images)
- OCR extraction of values
- Interactive charts showing trends over time
- Multiple parameters: WBC, RBC, Hemoglobin, Glucose, Cholesterol, etc.
- Normal range indicators

**User Value:**
- Visual identification of concerning trends
- Easy comparison across time periods
- Better patient education with visual aids
- Early warning for deteriorating conditions

**Technical Implementation (Fixture):**
- Recharts renders a sample time series from `src/data/dummyData.js`
- Upload is UI-only; PDF parsing and OCR (Tesseract) are not wired to this screen
- The banner on the screen states this in the UI, not just here

---

### 5. AI Diagnostic Insights
**What it does:**
- Analyzes all available patient data
- Provides evidence-based diagnostic suggestions
- Color-coded risk assessment
- Clinical recommendations
- Differential diagnosis ranking

**User Value:**
- Reduces diagnostic errors
- Considers all data points systematically
- Evidence-based recommendations
- Second opinion from AI

**Technical Implementation (Implemented):**
- RAG over patient records and uploaded PDFs (FAISS + MiniLM)
- Open-weight clinical model (`med` profile, default `medgemma:4b`) local via Ollama
- Confidence scoring per claim
- **Citation-or-drop:** a claim is shown only if the line it quotes exists on the page
  it cites (`services/citations.py`); verified claims carry an exact bounding box

---

## 👨‍💼 Patient Portal Features

### 1. Dashboard
**What it does:**
- Health overview with quick stats
- Upcoming medication reminders
- Next appointment information
- Health tips

**User Value:**
- Simple, non-technical interface
- Quick access to important info
- Personalized health guidance
- Peace of mind

---

### 2. Medication Management
**What it does:**
- Lists all prescribed medications
- Organized by time of day
- Detailed instructions for each medicine
- Purpose and dosage clearly explained
- Reminder settings

**User Value:**
- Reduces medication errors
- Improves adherence
- Clear understanding of purpose
- No confusion about timing

**Technical Implementation (Fixture):**
- The screen renders `src/data/dummyData.js` behind an amber fixture banner
- Push reminders, interaction checking and refill alerts are **not** implemented

---

### 3. AI Health Chatbot
**What it does:**
- Answers common health questions
- Medication schedule queries
- Diet and lifestyle advice
- Appointment information
- Natural language processing

**User Value:**
- 24/7 availability
- Instant answers to simple questions
- Reduces calls to doctor's office
- Empowers patients with information

**Technical Implementation (Implemented):**
- Local open-weight SLM (`slm` profile, default `qwen2.5:7b-instruct`)
- Patient-specific context assembled from MongoDB records
- Emergency-safe wording in the system prompt; refuses to diagnose

**Safety Features:**
- Emergency detection (chest pain, difficulty breathing)
- Automatic alert to doctor for concerning symptoms
- Clear disclaimer about limitations
- Direct emergency contact option

---

### 4. Diet & Nutrition Plan
**What it does:**
- Personalized diet recommendations
- Foods to eat and avoid
- Sample meal plans
- Dietary tips
- Downloadable plans

**User Value:**
- Clear dietary guidance
- Condition-specific recommendations
- Practical meal planning
- Improves treatment outcomes

**Technical Implementation (Fixture):**
- The plan text comes from `src/data/dummyData.js` behind an amber fixture banner
- Nutritionist templates and condition-based customisation are **not** implemented

---

### 5. Notifications & Reminders
**What it does:**
- Medication reminders
- Appointment reminders
- Test result availability alerts
- Health tips
- Customizable notification settings

**User Value:**
- Never miss medications
- Timely appointment attendance
- Stay informed about health status
- Reduce anxiety about forgetting

**Technical Implementation (Fixture):**
- The list renders `src/data/dummyData.js` behind an amber fixture banner
- Firebase Cloud Messaging, SMS fallback and scheduling are **not** implemented

---

## 🔐 Security & Privacy Features

### What is actually implemented
- Argon2 password hashing; JWT bearer tokens, 7-day expiry
- Role separation (doctor vs patient routes) with per-relationship access checks
- Secrets in environment variables only, `.env` gitignored — no patient data in the repo
- Local-first inference: with Ollama, document text never leaves the machine

### Not implemented — do not claim
- End-to-end encryption
- HIPAA compliance, or any certification or audit of any kind
- Immutable audit logs, per-field access trails, or data-anonymisation pipelines

### Patient Privacy
- Separate doctor and patient signup flows
- No third-party analytics or trackers in the frontend
- No data-export or delete-account endpoint yet; no anonymisation layer

> A production deployment would need encryption at rest, an audit trail and a
> retention policy. This is a prototype and says so.

---

## 🌍 Scalability Features

### Multi-Language Support (Planned)
- UI in 10+ languages
- Chatbot multilingual
- Voice transcription in regional languages

### Multi-Specialty Support
- Customizable fields per specialty
- Specialty-specific report templates
- Relevant test parameters per condition

### Integration Capabilities
- EHR system APIs
- Lab result auto-import
- Pharmacy integration
- Insurance claim integration

---

## 📊 Analytics & Insights (Planned)

### For Doctors
- Patient population health trends
- Treatment efficacy analysis
- Resource utilization
- Quality metrics

### For Healthcare Administrators
- Doctor productivity metrics
- Patient satisfaction scores
- Cost savings analysis
- System usage statistics

---

## 🎯 Unique Selling Points

1. **AI-First Design**: Not just AI features bolted on, but built around AI from the ground up
2. **Dual Portal**: Equally focused on doctor efficiency and patient empowerment
3. **Visual Intelligence**: Charts and visualizations make complex data simple
4. **Conversation-First**: Uses natural conversations as the foundation for documentation
5. **Production-Ready**: Professional UI, not a basic prototype

---

## 🚀 Future Roadmap

### Phase 2 (Next 3 months)
- Video consultations
- Prescription e-signing
- Lab order integration
- Appointment scheduling

### Phase 3 (6 months)
- Mobile apps (iOS/Android)
- Telemedicine integration
- Remote patient monitoring
- Wearable device integration

### Phase 4 (12 months)
- AI-powered diagnosis (FDA approval process)
- Predictive health analytics
- Population health management
- Research data contribution

---

## 💻 Technical Architecture Highlights

### Frontend
- **React 18**: Modern hooks, concurrent rendering
- **Tailwind CSS**: Utility-first styling, responsive design
- **Recharts**: Interactive, accessible charts
- **React Router v6**: Modern routing with nested routes
- **Vite**: Lightning-fast dev server and builds

### Backend (implemented)
- **FastAPI**: async Python web framework
- **MongoDB**: users, patients, appointments, chat history
- **FAISS + MiniLM**: per-document vector index, in memory
- **Local disk or Cloudflare R2**: document storage, chosen at runtime

### AI stack (implemented)
- **Ollama** (OpenAI-compatible): `qwen2.5:7b-instruct` for extraction/chat,
  `medgemma:4b` for clinical reasoning — no API key
- **faster-whisper**: local speech-to-text; **pyttsx3**: local text-to-speech
- **citations.py**: deterministic claim verification (no model)
- **bbox.py**: exact word-rect geometry for evidence highlighting

### DevOps (implemented)
- `scripts/preflight.py` + `/healthz`: readiness checks before a demo
- Offline tests: `tests/test_citations.py`, `tests/test_bboxes.py`
- **Not implemented:** Docker, Kubernetes, CI/CD, monitoring

---

## 📈 What we would measure (not measured yet)

No efficiency or accuracy figure in this document is a measured result, so none are
claimed. The honest position:

- **Time saved per consultation** — needs a timed study with doctors; not run.
- **Diagnostic accuracy delta** — needs a labelled set and clinician review; not run.
- **Cost impact** — not modelled.

What *can* be measured today, offline, and is: how many offered claims survive
verification, and how often the page number had to be corrected. That number comes out
of `citations.CitationReport` on every document and is printed in the UI.

---

EUNOCURA is a document-intelligence harness with two portals on top, not a certified
clinical system. The parts that decide what a doctor sees are deterministic, tested, and
small enough to read in one sitting — which is the actual claim being made here.

