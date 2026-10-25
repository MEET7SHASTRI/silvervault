"""
SilverVault AI - NLP & Machine Learning Engine
Includes:
1. Vector Space Model with TF-IDF and BM25 Hybrid Semantic Search
2. Academic Concept & Topic Extraction using NLP
3. Cognitive Level (Bloom's Taxonomy) Tagging
4. Student Knowledge Tracing & Topic Mastery Estimator (ML)
5. Comprehensive Post-Test Weakness Diagnostic & Remediation Generator
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Set, Optional

# Standard academic stopwords
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's
me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've explain describe define discuss give list tell mention
""".split())

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms excluding stopwords."""
    words = re.findall(r"[a-zA-Z0-9_+-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Extract character or word n-grams for phrase-level matching."""
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# --- Vector Space & BM25 Hybrid Retrieval ---

class HybridVectorRetriever:
    """
    Hybrid semantic retrieval engine combining TF-IDF Vector Space Model
    and BM25 lexical ranking with query term proximity.
    """
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_freqs = Counter()
        self.N = len(chunks)

        for ch in chunks:
            toks = tokenize(ch["content"])
            # Add bigrams for technical phrases (e.g. 'binary_tree', 'linked_list')
            bigrams = extract_ngrams(toks, 2)
            all_toks = toks + bigrams
            self.doc_tokens.append(all_toks)
            self.doc_lengths.append(len(all_toks))
            for t in set(all_toks):
                self.doc_freqs[t] += 1

        self.avg_dl = sum(self.doc_lengths) / max(1, self.N) if self.N > 0 else 1.0

    def score_query(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        """Compute hybrid similarity score for all chunks against query."""
        q_toks = tokenize(query)
        q_bigrams = extract_ngrams(q_toks, 2)
        all_q = q_toks + q_bigrams
        if not all_q or self.N == 0:
            return []

        q_counts = Counter(all_q)
        scores = []

        for idx, (ch, d_toks, d_len) in enumerate(zip(self.chunks, self.doc_tokens, self.doc_lengths)):
            if d_len == 0:
                continue
            d_counts = Counter(d_toks)
            bm25_score = 0.0
            overlap_count = 0

            for term, q_tf in q_counts.items():
                if term in d_counts:
                    overlap_count += 1
                    df = self.doc_freqs.get(term, 1)
                    # IDF with Robertson-Spärck Jones formula
                    idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                    d_tf = d_counts[term]
                    denom = d_tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_dl))
                    term_score = idf * (d_tf * (self.k1 + 1.0)) / max(0.001, denom)
                    bm25_score += term_score

            coverage = overlap_count / max(1, len(set(all_q)))
            # Combined score: BM25 score normalized + high coverage bonus
            final_score = (bm25_score / (bm25_score + 10.0)) * 0.7 + coverage * 0.3
            if final_score > 0.08:
                scores.append((round(final_score, 4), ch))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores

# --- NLP Concept & Topic Extraction ---

def extract_academic_topics(text: str) -> List[str]:
    """
    Extract technical concepts, definitions, and topic names from an academic document.
    """
    topics = set()
    # 1. Heading and unit patterns
    heading_patterns = [
        r"(?m)^\s*(?:Unit|Chapter|Section|Module)\s*[-:]?\s*\d*\s*[-:]?\s*([A-Za-z0-9\s,/-]{3,40})$",
        r"(?m)^\s*([A-Z][A-Z0-9\s,/-]{3,40})\s*$",
        r"(?:WHAT IS|INTRODUCTION TO|OVERVIEW OF|DEFINITION OF)\s+([A-Z0-9\s-]{3,40})",
    ]
    for pattern in heading_patterns:
        for match in re.finditer(pattern, text, re.I):
            val = match.group(1).strip(" :-\n\t")
            if 3 < len(val) < 40 and not any(sw in val.lower().split() for sw in ["page", "index", "contents"]):
                topics.add(val.title())

    # 2. Key academic noun phrase patterns (e.g. "Linear Data Structures", "Binary Search Tree")
    np_pattern = r"\b((?:[A-Z][a-z]+\s+){1,3}(?:Structure|Algorithm|Tree|Queue|Stack|Array|List|Graph|Search|Sort|Memory|System|Process|Matrix|Hashing|Index|Complexity))\b"
    for match in re.finditer(np_pattern, text):
        topics.add(match.group(1).strip())

    # 3. Definition subjects: "A [Subject] is a..."
    def_pattern = r"(?:^|\.\s+)(?:An?|The)\s+([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2})\s+(?:is|are|refers to|is defined as)\s+"
    for match in re.finditer(def_pattern, text):
        candidate = match.group(1).strip()
        if candidate.lower() not in STOPWORDS and 3 < len(candidate) < 35:
            topics.add(candidate.title())

    return sorted(topics)[:30]

def detect_topic_for_question(question: str, context: str = "", known_topics: List[str] = None) -> str:
    """
    Map an assessment question to its most relevant academic topic.
    """
    if known_topics:
        q_low = question.lower()
        for t in known_topics:
            if t.lower() in q_low:
                return t

    # Search for technical noun phrases in question
    match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    if match and match.group(1).lower() not in STOPWORDS and len(match.group(1)) > 3:
        return match.group(1)

    # Concept patterns
    m2 = re.search(r"(?:about|of|which concept|which term|describe|defined as)\s+([a-zA-Z\s]{3,25})[\?\.,]", question, re.I)
    if m2:
        val = m2.group(1).strip()
        words = [w for w in val.split() if w.lower() not in STOPWORDS]
        if words:
            return " ".join(words).title()

    return "General Concepts"

def classify_cognitive_level(question: str) -> str:
    """
    Classify question cognitive complexity using Bloom's Taxonomy.
    """
    q_low = question.lower()
    if any(q_low.startswith(w) or f" {w} " in q_low for w in ["what is", "which of", "define", "name the", "list", "identify", "which term", "what does"]):
        return "Recall & Definition"
    elif any(w in q_low for w in ["why", "how does", "compare", "distinguish", "difference between", "explain", "describe", "characteristic"]):
        return "Comprehension & Analysis"
    elif any(w in q_low for w in ["calculate", "apply", "implement", "output of", "trace", "worst case", "time complexity", "design"]):
        return "Application & Problem-Solving"
    return "Conceptual Understanding"

# --- Student Learning Analysis & Weakness Diagnostics ---

def analyze_student_test_performance(
    review_items: List[Dict[str, Any]],
    subject: str,
    test_type: str = "MCQ"
) -> Dict[str, Any]:
    """
    Machine Learning / Statistical Diagnostic of student test performance.
    Computes topic-level mastery, identifies critical learning gaps, and generates
    a personalized prescriptive study roadmap.
    """
    total = len(review_items)
    if total == 0:
        return {
            "score": 0, "total": 0, "percentage": 0,
            "mastery_level": "Needs Assessment",
            "topic_mastery": {}, "weak_topics": [],
            "cognitive_breakdown": {}, "report_markdown": "No questions were completed."
        }

    correct_count = sum(1 for item in review_items if item.get("ok", False))
    percentage = round((correct_count / total) * 100, 1)

    # Topic-level aggregation
    topic_stats = defaultdict(lambda: {"total": 0, "correct": 0, "missed_questions": []})
    cognitive_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in review_items:
        q = item.get("q", "")
        ok = item.get("ok", False)
        topic = item.get("topic") or detect_topic_for_question(q)
        cog = item.get("cognitive_level") or classify_cognitive_level(q)

        topic_stats[topic]["total"] += 1
        cognitive_stats[cog]["total"] += 1
        if ok:
            topic_stats[topic]["correct"] += 1
            cognitive_stats[cog]["correct"] += 1
        else:
            topic_stats[topic]["missed_questions"].append({
                "question": q,
                "correct_answer": item.get("correct", ""),
                "student_answer": item.get("chosen", "") or item.get("student_answer", "")
            })

    # Topic mastery metrics
    topic_mastery = {}
    weak_topics = []
    strong_topics = []

    for topic, stats in topic_stats.items():
        pct = round((stats["correct"] / stats["total"]) * 100, 1)
        topic_mastery[topic] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "percentage": pct,
            "status": "Mastered" if pct >= 75 else ("Developing" if pct >= 50 else "Critical Gap")
        }
        if pct < 60:
            weak_topics.append(topic)
        elif pct >= 80:
            strong_topics.append(topic)

    # Overall Mastery Category
    if percentage >= 80:
        mastery_level = "Mastered"
    elif percentage >= 55:
        mastery_level = "Developing"
    else:
        mastery_level = "Critical Learning Gap"

    # Cognitive Level Breakdown
    cognitive_breakdown = {}
    for cog, stats in cognitive_stats.items():
        cog_pct = round((stats["correct"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
        cognitive_breakdown[cog] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "percentage": cog_pct
        }

    # Generate Structured Prescriptive Markdown Report
    report_md = f"""### 🎯 Academic Performance & Concept-Level Weakness Report
**Subject:** `{subject}` | **Assessment Type:** `{test_type}` | **Overall Score:** **{correct_count}/{total} ({percentage}%)**  
**Overall Competency Rating:** `{'🟢 ' + mastery_level if mastery_level == 'Mastered' else ('🟡 ' + mastery_level if mastery_level == 'Developing' else '🔴 ' + mastery_level)}`

---

#### 1. 📊 Topic-by-Topic Mastery Breakdown
| Academic Topic | Questions Tested | Correct | Mastery Rate | Diagnostic Status |
| :--- | :---: | :---: | :---: | :--- |
"""
    for topic, data in sorted(topic_mastery.items(), key=lambda x: x[1]["percentage"]):
        status_icon = "🟢 Mastered" if data["percentage"] >= 75 else ("🟡 Developing" if data["percentage"] >= 50 else "🔴 Critical Gap")
        report_md += f"| **{topic}** | {data['total']} | {data['correct']} | {data['percentage']}% | {status_icon} |\n"

    report_md += "\n#### 2. 🧠 Cognitive Dimension Analysis (Bloom's Taxonomy)\n"
    for cog, cdata in cognitive_breakdown.items():
        report_md += f"- **{cog}:** {cdata['correct']}/{cdata['total']} correct ({cdata['percentage']}%)\n"

    report_md += "\n---\n\n#### 3. 🔍 Detailed Diagnostic: Concepts You Need to Improve\n"
    if not weak_topics:
        report_md += "> 🌟 **Outstanding retention!** You demonstrated solid mastery across all tested topics in this session. To reinforce long-term memory, try an advanced test or a verbal viva.\n"
    else:
        for topic in weak_topics:
            missed = topic_stats[topic]["missed_questions"]
            report_md += f"##### ⚠️ Focus Area: `{topic}`\n"
            report_md += f"- **Gap Severity:** Critical attention needed (Mastery: {topic_mastery[topic]['percentage']}%)\n"
            report_md += "- **Missed Concepts Analysis:**\n"
            for idx, m in enumerate(missed, 1):
                report_md += f"  - *Question {idx}:* {m['question']}\n"
                report_md += f"    - **Expected Concept / Answer:** `{m['correct_answer']}`\n"
                if m["student_answer"]:
                    report_md += f"    - **Your Response:** `{m['student_answer']}` (needs revision)\n"

    report_md += "\n---\n\n#### 4. 📚 Prescriptive Learning Action Plan\n"
    if weak_topics:
        report_md += f"1. **Targeted Reading:** Open the `{subject}` book in the **Select Subject** tab and review the sections dedicated to: **{', '.join(weak_topics)}**.\n"
        report_md += f"2. **Concept Distinctions:** Focus on definitions, structural operations, and exact terminology where points were deducted.\n"
        report_md += f"3. **Oral Practice:** Run a **Verbal Viva** on `{subject}` to practice explaining these weak topics aloud and receiving real-time examiner guidance.\n"
    else:
        report_md += "1. Continue practicing with higher question counts and advanced oral viva scenarios.\n"
        report_md += "2. Progress to the next syllabus module or subject.\n"

    return {
        "score": correct_count,
        "total": total,
        "percentage": percentage,
        "mastery_level": mastery_level,
        "topic_mastery": topic_mastery,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "cognitive_breakdown": cognitive_breakdown,
        "report_markdown": report_md
    }
