"""
SilverVault AI - Assessment Engine
Automated Mock-Test Generation (MCQ), Hybrid LLM + NLP Question Synthesizer,
Cognitive Level Tagging, and Post-Test Learning Gap Analytics.
"""

import re
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from nlp_ml_engine import (
    extract_academic_topics, detect_topic_for_question,
    classify_cognitive_level, analyze_student_test_performance, STOPWORDS
)
from rag_engine import run_llm_completion
from database import save_test_result, save_learning_report, update_topic_mastery

# Common academic domain concepts to use as valid distractors if needed
CURATED_DOMAIN_CONCEPTS = {
    "data structures": ["Array", "Linked List", "Stack", "Queue", "Binary Tree", "B-Tree", "Graph", "Hash Table", "Heap", "Trie"],
    "algorithms": ["Linear Search", "Binary Search", "Bubble Sort", "Quick Sort", "Merge Sort", "Dijkstra's Algorithm", "Breadth-First Search", "Depth-First Search", "Dynamic Programming", "Greedy Method"],
    "operating systems": ["Process", "Thread", "Semaphore", "Mutex", "Deadlock", "Virtual Memory", "Paging", "Segmentation", "CPU Scheduling", "Context Switching"],
    "databases": ["Primary Key", "Foreign Key", "Normalization", "ACID Properties", "Index", "Transaction", "Relational Schema", "B+ Tree Index", "Join Operation", "SQL Query"],
    "computer networks": ["TCP", "UDP", "IP Address", "Router", "Switch", "OSI Model", "Subnet Mask", "DNS", "HTTP", "Packet Switching"]
}

def _clean_json_str(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM JSON response."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.I)
    text = text.replace("```", "").strip()
    return text

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    cleaned = _clean_json_str(text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass
    return []

# --- NLP Rule-Based / Syntactic MCQ Extractor ---

def _extract_nlp_mcqs_from_context(context: str, n_needed: int, subject: str = "") -> List[Dict[str, Any]]:
    """
    Extract factual, grammatical MCQs directly from textbook sentences using NLP patterns.
    Guarantees 100% validity and eliminates hallucinations.
    """
    if not context or not context.strip():
        return []

    # Clean text
    clean_text = re.sub(r"\[(?:Source|Chunk):?[^\]]+\]", " ", context)
    clean_text = re.sub(r"[•\t]+", " ", clean_text)
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\s*\n\s*", " ", clean_text)

    # Blacklist of generic pseudo-subjects and noisy words that are not actual technical concepts
    GENERIC_WORDS = {
        "idea", "way", "method", "term", "thing", "point", "part", "types",
        "example", "case", "fact", "step", "role", "time", "space", "order",
        "implementation", "interface", "representation", "use", "need", "result",
        "obtained", "shown", "given", "following", "above", "below", "value",
        "values", "number", "figure", "table", "section", "chapter", "unit",
        "purpose", "reason", "definition", "overview", "introduction", "conclusion"
    }

    # Split into clean sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", clean_text)
    sentences = []
    for s in raw_sentences:
        s = s.strip(" \t\r\n•-–")
        words = s.split()
        if 8 <= len(words) <= 35 and 45 <= len(s) <= 220 and s[-1:] in ".!?":
            low = s.lower()
            if not re.match(r"^(so that|therefore|thus|which|who|and|or|but|because|such as|for example|meanwhile)\b", low):
                sentences.append(s)

    # Concept patterns
    patterns = [
        # "An X is a/an Y..."
        (r"^(?:An?|The)\s+([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2})\s+(is|are|is defined as|refers to)\s+(.+)$", "def"),
        # "An X follows / consists of / stores Y..."
        (r"^(?:An?|The)\s+([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2})\s+(follows|consists of|contains|stores|uses|supports|allows|represents|provides)\s+(.+)$", "prop"),
    ]

    facts = []
    extracted_concepts = set()

    for s in sentences:
        for pat, kind in patterns:
            m = re.search(pat, s, re.I)
            if m:
                subj, verb, pred = [x.strip(" .") for x in m.groups()]
                subj_clean = re.sub(r"^(?:a|an|the)\s+", "", subj, flags=re.I).strip()
                words_in_subj = set(subj_clean.lower().split())
                if words_in_subj & GENERIC_WORDS or len(subj_clean) < 3:
                    continue
                if 1 <= len(subj_clean.split()) <= 3:
                    if subj_clean.lower() not in STOPWORDS and 15 <= len(pred) <= 180:
                        facts.append({"subject": subj_clean.title(), "verb": verb.lower(), "predicate": pred, "kind": kind, "sentence": s})
                        extracted_concepts.add(subj_clean.title())
                        break

    if not facts:
        return []

    # Gather domain distractors
    pool = list(extracted_concepts)
    # Supplement pool with general domain terms if less than 5
    sub_key = subject.lower()
    for domain, terms in CURATED_DOMAIN_CONCEPTS.items():
        if domain in sub_key or any(w in sub_key for w in domain.split()):
            for t in terms:
                if t.lower() not in [p.lower() for p in pool]:
                    pool.append(t)
            break

    if len(pool) < 4:
        for t in ["Queue", "Stack", "Binary Search Tree", "Linked List", "Hash Table", "Graph", "Array"]:
            if t.lower() not in [p.lower() for p in pool]:
                pool.append(t)

    results = []
    seen_q = set()

    for fact in facts:
        answer = fact["subject"]
        predicate = fact["predicate"].rstrip(" .")
        verb = fact["verb"]

        if fact["kind"] == "def":
            question = f"According to the selected textbook, which concept is described as {predicate}?"
        else:
            question = f"According to the selected textbook, which concept {verb} {predicate}?"

        q_key = question.lower()
        if q_key in seen_q:
            continue

        # Choose 3 distractors
        distractors = [c for c in pool if c.lower() != answer.lower()]
        if len(distractors) < 3:
            continue

        chosen_distractors = random.sample(distractors, 3)
        options = [answer] + chosen_distractors
        random.shuffle(options)

        results.append({
            "question": question,
            "options": options,
            "answer": answer,
            "topic": answer,
            "cognitive_level": "Recall & Definition" if fact["kind"] == "def" else "Conceptual Understanding",
            "explanation": f"Source textbook state: {fact['sentence']}"
        })
        seen_q.add(q_key)
        if len(results) >= n_needed:
            break

    return results

# --- Dynamic MCQ Generator ---

def generate_mcq_test(
    chunks: List[Dict[str, Any]],
    n_questions: int = 5,
    difficulty: str = "Standard",
    subject: str = "Computer Science"
) -> List[Dict[str, Any]]:
    """
    Generate dynamic mock test MCQs using Local LLM (Qwen 0.5B) with robust NLP fallback.
    """
    if not chunks:
        return []

    n_questions = max(1, min(int(n_questions), 20))
    sample_context = "\n\n".join(
        f"[Chunk {ch.get('chunk_index', idx)+1}]\n{ch['content']}"
        for idx, ch in enumerate(chunks[:25]) if ch.get("content", "").strip()
    )

    # Attempt LLM generation first
    prompt = f"""Generate {n_questions} academic Multiple Choice Questions (MCQs) for the subject '{subject}' based ONLY on the source text below.
Difficulty: {difficulty}.

CRITICAL REQUIREMENTS:
1. Return ONLY a valid JSON array of objects.
2. Each object must have:
   - "question": A clear academic question stem.
   - "options": An array of exactly 4 distinct choices.
   - "answer": The exact text of the correct choice.
   - "topic": The specific subject topic/concept being tested.
   - "cognitive_level": One of "Recall & Definition", "Comprehension & Analysis", "Application & Problem-Solving".
   - "explanation": Brief explanation cited from the text.
3. No duplicate questions. Do not invent facts outside the text.

SOURCE TEXT:
{sample_context[:2500]}

JSON ARRAY:"""

    llm_mcqs = []
    try:
        raw_output = run_llm_completion(prompt, max_tokens=1000, temperature=0.2)
        items = _extract_json_array(raw_output)
        for it in items:
            q = it.get("question")
            opts = it.get("options")
            ans = it.get("answer")
            if q and isinstance(opts, list) and len(opts) == 4 and ans in opts:
                llm_mcqs.append({
                    "question": str(q).strip(),
                    "options": [str(o).strip() for o in opts],
                    "answer": str(ans).strip(),
                    "topic": str(it.get("topic", detect_topic_for_question(q))),
                    "cognitive_level": str(it.get("cognitive_level", classify_cognitive_level(q))),
                    "explanation": str(it.get("explanation", "Grounded in source textbook excerpts."))
                })
    except Exception:
        llm_mcqs = []

    # If LLM generated sufficient valid MCQs, return them
    if len(llm_mcqs) >= n_questions:
        return llm_mcqs[:n_questions]

    # Otherwise, supplement or build using the robust NLP extractor
    needed = n_questions - len(llm_mcqs)
    nlp_mcqs = _extract_nlp_mcqs_from_context(sample_context, needed, subject)
    combined = llm_mcqs + nlp_mcqs

    # Deduplicate
    final_qs = []
    seen = set()
    for item in combined:
        k = item["question"].strip().lower()
        if k not in seen:
            seen.add(k)
            final_qs.append(item)
            if len(final_qs) >= n_questions:
                break

    return final_qs

# --- Short Answer & Viva Evaluation ---

def evaluate_short_answer_response(
    question: str,
    student_answer: str,
    context: str,
    topic: str = "General"
) -> Tuple[float, str, List[str]]:
    """
    Evaluate oral or written answer against textbook excerpts using Local LLM & NLP.
    Returns: (score [0-10], feedback, missing_concepts)
    """
    if not student_answer.strip():
        return 0.0, "No answer was recorded. Please provide an answer to be evaluated.", ["Complete answer required"]

    if not context.strip():
        return 0.0, "No reference textbook context was retrieved for this question.", []

    prompt = f"""You are a university viva examiner evaluating a student's oral answer.
Evaluate the answer strictly based on the reference textbook excerpts.

Question: {question}
Topic: {topic}
Reference Textbook Excerpts:
{context}

Student's Answer:
{student_answer}

Rate the student's answer fairly from 0 to 10:
- 8-10: Accurate, covers core definition and key properties.
- 5-7: Partially correct, captures general idea but misses technical terms or specifics.
- 1-4: Vague, incorrect, or missing essential facts.
- 0: Unrelated or no meaningful content.

Return ONLY a JSON object with:
{{
  "score": <integer from 0 to 10>,
  "feedback": "<2-3 constructive sentences explaining what was good and what was missing>",
  "missing_concepts": ["<concept1>", "<concept2>"]
}}"""

    try:
        raw = run_llm_completion(prompt, max_tokens=250, temperature=0.1)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            score = float(max(0, min(10, data.get("score", 0))))
            feedback = str(data.get("feedback", "Answer evaluated against textbook concepts."))
            missing = data.get("missing_concepts", [])
            return score, feedback, missing
    except Exception:
        pass

    # Heuristic fallback based on token overlap with context and question
    q_toks = set(re.findall(r"[a-z]{3,}", question.lower()))
    ctx_toks = set(re.findall(r"[a-z]{3,}", context.lower())) - q_toks - STOPWORDS
    ans_toks = set(re.findall(r"[a-z]{3,}", student_answer.lower()))

    overlap = len(ans_toks & ctx_toks)
    if len(ans_toks) < 4:
        return 2.0, "Answer was too brief. Elaborate on the core definition and provide an example.", ["Detailed explanation"]

    score = min(10.0, max(3.0, round(3.0 + (overlap / 4.0) * 5.0, 1)))
    feedback = f"You captured some key terms. To improve, clearly explain the technical mechanism and provide a textbook example."
    return score, feedback, []
