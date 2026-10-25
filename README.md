# SilverVault AI: A Subject-Aware Retrieval-Augmented Generation Framework for Academic Assistance and Intelligent Voice-Based Assessment

**Keywords:** Retrieval-Augmented Generation, Academic Question Answering, Semantic Search, Lightweight Language Models, Automated Assessment, Voice-Based Viva, Student Performance Analytics, Document-Grounded AI.

---

## Abstract
The increasing availability of digital academic resources has created a need for intelligent learning systems capable of providing contextually relevant explanations, personalized practice, and efficient knowledge assessment. However, conventional conversational AI systems may generate responses that extend beyond prescribed learning materials, while generic assessment tools often lack the ability to identify concept-level weaknesses.

To address these limitations, **SilverVault AI** proposes a subject-aware academic assistance framework based on Retrieval-Augmented Generation (RAG), integrating document-grounded question answering, automated mock-test generation, performance analytics, and an interactive voice-based viva assessment module. The system organizes academic resources according to semester and subject, enabling contextual retrieval from administrator-provided reference documents and student-specific PDF uploads. A vector-based retrieval mechanism identifies semantically relevant passages, which are supplied to a lightweight language model to generate answers grounded in the selected academic material. The framework incorporates context-bound response generation to reduce unsupported outputs when the requested information is unavailable in the source document. Furthermore, the mock-assessment module generates subject-oriented questions and evaluates student responses to identify weaker concepts and maintain a record of learning performance.

The proposed voice-based assessment component extends this functionality by converting spoken responses into text, comparing the extracted content against relevant reference concepts, and delivering automated feedback through speech synthesis. To support resource-constrained deployment, the architecture emphasizes locally executable models and modular components, reducing dependence on external inference services. Student-specific document isolation and persistent academic histories are incorporated to support personalized learning and controlled access to uploaded resources. Overall, SilverVault AI demonstrates how retrieval-grounded language generation, lightweight inference, and multimodal interaction can be combined into a unified academic support environment.

---

## System Architecture

```
                                  +---------------------------------------+
                                  |         Student Web Interface         |
                                  |     (Streamlit UI + Web Speech API)   |
                                  +-------------------+-------------------+
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                   |                                   |
                  v                                   v                                   v
        +-------------------+               +-------------------+               +-------------------+
        |  Ask Your Book    |               |   MCQ Mock Exam   |               | Continuous Viva   |
        |  (Grounded RAG)   |               | (Dynamic Question)|               | (Hands-free Voice)|
        +---------+---------+               +---------+---------+               +---------+---------+
                  |                                   |                                   |
                  +-----------------+                 |                 +-----------------+
                                    |                 |                 |
                                    v                 v                 v
                               +---------------------------------------------+
                               |              Core AI Engine                 |
                               |  - Hybrid Vector Retriever (TF-IDF + BM25)  |
                               |  - Local Qwen 0.5B LLM (llama-cpp-python)   |
                               |  - NLP Concept & Syntactic Pattern Analyzer |
                               |  - Bayesian Knowledge State Mastery Tracker |
                               +----------------------+----------------------+
                                                      |
                                                      v
                               +---------------------------------------------+
                               |        Persistent Storage Layer             |
                               |  - SQLite (WAL Mode, Salted PBKDF2 Auth)    |
                               |  - Chunk Index & Metadata Store             |
                               |  - Learning-Gap Diagnostic Reports          |
                               +---------------------------------------------+
```

---

##  NLP, Machine Learning & Deep Learning Implementation

1. **Deep Learning (Local LLM Inference)**:
   - Uses `Qwen 2.5 0.5B Instruct` in quantized GGUF format via `llama-cpp-python`.
   - Local CPU/GPU inference with zero cloud dependency.
   - Context-bound prompt engineering ensures strict grounding: answers are derived exclusively from textbook passages, preventing academic hallucinations.

2. **Natural Language Processing (NLP)**:
   - **Page-Aware Document Ingestion**: PyMuPDF (`pymupdf`) extracts page boundaries and paragraph structures.
   - **Recursive Chunking with Overlap**: Preserves sentence boundaries, headers, and context continuity.
   - **Concept & Terminology Extraction**: Regex and syntactic rule engines identify key definitions, structural properties, and domain entities.
   - **Cognitive Level Classification**: Automatically categorizes assessment questions using Bloom's Taxonomy (*Recall & Definition*, *Comprehension & Analysis*, *Application & Problem-Solving*).

3. **Machine Learning & Information Retrieval**:
   - **Hybrid Semantic Search**: Combines TF-IDF vector space modeling with BM25 lexical ranking (Robertson-Spärck Jones IDF formula) for high retrieval precision.
   - **Bayesian Knowledge Tracing (BKT)**: Dynamically updates student topic mastery states using exponential moving average updates:
     $$\text{Mastery}_{t} = (1 - \alpha) \cdot \text{Mastery}_{t-1} + \alpha \cdot \text{Observation}_t$$
   - **Post-Test Weakness Diagnostics**: Aggregates missed questions into topic clusters, quantifies learning gap severity, and outputs prescriptive reading roadmaps.

---

## 🎙️ Continuous Voice Viva Examiner

- **Continuous Assessment Loop**:
  Unlike traditional fixed-count quizzes, the viva runs continuously:
  1. The AI Examiner selects a textbook concept and formulates an oral question.
  2. The browser speaks the question aloud using speech synthesis.
  3. The student speaks their answer into the microphone.
  4. Web Speech Recognition captures live speech with real-time waveform visualization.
  5. Upon a natural pause (~2.2s) or manual confirmation, the response is sent to the local model.
  6. The response is evaluated against retrieved reference passages, generating a score (0-10) and examiner feedback.
  7. The system dynamically queries the next textbook topic and asks the next question.
  8. **The exam continues indefinitely until the student clicks "Stop Viva and View Report".**
  9. Upon conclusion, a comprehensive diagnostic report is compiled and saved.

---

## How to Run

### One-Click Launcher (Recommended for Windows)
Double-click `SilverVault_AI.exe`

---

## Security & Privacy
- **Student Data Isolation**: Documents and chat histories are scoped strictly to the student's authenticated ID.
- **PBKDF2 Password Hashing**: Passwords stored with cryptographic salts and 240,000 iterations.
- **Local Model Execution**: Local inference ensures academic notes and assessment transcripts never leave your machine.
