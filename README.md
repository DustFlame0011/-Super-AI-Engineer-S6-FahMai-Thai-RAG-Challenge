# 🏪 FahMai RAG Challenge — Super AI Engineer Season 6

> **Competition:** [Super AI Engineer S6 — FahMai RAG Challenge (Level 1)](https://www.kaggle.com/competitions/super-ai-engineer-s-6-fah-mai-rag-challenge-level-1)
> **Platform:** Kaggle
> **Type:** Retrieval-Augmented Generation (RAG) · Multiple Choice QA · Thai NLP
> **Final Score:** Public `0.71` · Private `0.82` · Leaderboard `+144` positions
<img width="1195" height="173" alt="Image" src="https://github.com/user-attachments/assets/68927362-039c-4e84-b4f0-23c36ebfc4a3" />
---

## 📌 Overview

<img width="2752" height="1536" alt="Image" src="https://github.com/user-attachments/assets/d197dc4a-4c0a-4b6a-8461-438e41e400b3" />

This project is my participation in the **Super AI Engineer Season 6 Hackathon**, specifically the **FahMai RAG Challenge (Level 1)**. The task was to build a RAG system that answers **100 multiple-choice questions** (10 choices each) about a fictional Thai electronics store called **FahMai (ฟ้าใหม่)**, using a knowledge base of Thai-language Markdown documents covering products, store policies, and branch information.

Each question includes:
- **Choices 1–8:** Content-specific answers
- **Choice 9:** "No data available in the knowledge base"
- **Choice 10:** "This question is not related to FahMai store"

**Evaluation Metric:** Accuracy (% correct out of 100 questions)
- Public Leaderboard: 60% of questions
- Private Leaderboard: 40% of questions

---

## 🏗️ System Architecture

<img width="1407" height="768" alt="Image" src="https://github.com/user-attachments/assets/86812856-4bea-4f01-acd1-212b407100f7" />

---

## 🛠️ Tech Stack

| Component | Tool / Model |
|---|---|
| Embedding | `BAAI/bge-m3` (1024-dim multilingual) |
| Sparse Retrieval | BM25 via `rank-bm25` |
| Thai Tokenization | `pythainlp` (newmm engine) |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | ThaiLLM API — `typhoon` + `openthaigpt` |
| Voting | Majority vote across 2 models |
| Runtime | Google Colab |

---

## 📈 Score Progression

| Version | Key Change | Public Score |
|---|---|---|
| v1 | Starter kit (MiniLM + character chunking) | 0.06 |
| v2 | Fixed prompt bug (context not extracted) | 0.08 |
| v3 | BGE-M3 + markdown chunking + fixed TOP_K | 0.53 |
| v4 | Re-embed after chunk rebuild + TOP_K=8 | 0.70 |
| v5 | Query expansion + majority vote | **0.71** ✅ |

**🏁 Final Results**

| Metric | Score |
|---|---|
| Public Leaderboard | **0.71** |
| Private Leaderboard | **0.82** |
| Leaderboard Movement | **+144 positions** |

---

## 🔄 Attempt-by-Attempt Breakdown

<img width="2593" height="400" alt="Image" src="https://github.com/user-attachments/assets/e7119ec9-54ad-4fc1-8750-14872d64deee" />

### Attempt 1 — Public Score: 0.06
Started with the Starter Kit provided by Nutchanon Yongsatianchot, Ph.D., which used `paraphrase-multilingual-MiniLM-L12-v2` for embeddings and fixed-size character chunking (512 chars, 128 overlap). The pipeline ran without errors, but the score was 0.06 — far below the 0.73 baseline. At this point it was unclear where the problem was.

### Attempt 2 — Public Score: 0.08
Investigated the low score and discovered two silent bugs. First, `build_prompt()` was passing raw Python dict objects `{'text': ..., 'source': ...}` to the LLM instead of extracting the actual text — so the AI was reading code structure, not Thai content. Second, the `dense_retrieve()` function was returning all 1,053 chunk indices instead of only the top-5, making the prompt 562,059 characters long (≈187,000 tokens). Fixed both issues, but the score only improved slightly to 0.08, indicating retrieval quality was still poor.

### Attempt 3 — Public Score: 0.53
Made three changes at once. Replaced `MiniLM` with `BAAI/bge-m3`, a much stronger multilingual embedding model with 1024 dimensions that handles Thai significantly better. Switched from random character-based chunking to Markdown-aware chunking (`chunk_by_markdown()`), which splits documents on header boundaries (`## ...`) to preserve semantic structure. Also added a metadata prefix to each chunk so the LLM always knows which document and category the text came from. Score jumped from 0.08 to 0.53.

### Attempt 4 — Public Score: 0.70
Hit an `IndexError: list index out of range` crash. The cause was that after rebuilding chunks with the new Markdown-aware method, the chunk count changed but `chunk_embeddings` was still the old matrix from the previous run — the two fell out of sync. Fixed by always re-running `embed_model.encode()` and rebuilding the BM25 index immediately after any change to chunks. Also increased `TOP_K` from 5 to 8 to give the LLM more context per question. Score improved to 0.70.

### Attempt 5 (Final) — Public Score: 0.71 · Private Score: 0.82
Added two final improvements. First, **query expansion** — instead of retrieving using only the question, the query was expanded to include the first 4 content-specific answer choices (1–8). This gave BM25 more keywords to match against, especially for product names and specs. Second, **majority voting** — instead of relying on a single Thai LLM (`typhoon`), both `typhoon` and `openthaigpt` were queried for each question and the most common answer was selected. Final public score: 0.71. After the competition closed, the private leaderboard revealed a score of 0.82, with a leaderboard jump of +144 positions.

<img width="861" height="73" alt="Image" src="https://github.com/user-attachments/assets/4f46ea3d-7ce8-446f-967b-ef7e14f53be4" />

---

## 🐛 Problems Encountered & Solutions

### 1. Prompt Was 562,059 Characters Long (187K Tokens)
**Problem:** The `build_prompt()` function received the entire list of 1,053 chunks instead of the top-5 retrieved ones. This happened because `dense_retrieve()` returned a NumPy array of all indices without slicing, causing `[chunks[i] for i in idx]` to iterate over all 1,053 values.

**Root Cause:** The `k` parameter in `dense_retrieve()` was being ignored — `np.argsort(scores)[::-1][:k]` sliced correctly, but `k` defaulted to a module-level `TOP_K` variable that had been overwritten elsewhere.

**Fix:** Hardcoded `k=5` (later `k=8`) as the default argument directly in the function signature and always cast indices to `int` to avoid NumPy indexing behavior.

```python
def dense_retrieve(query, chunk_embs, k=8):
    ...
    top_idx = np.argsort(scores)[::-1][:k]  # explicit slice
    return top_idx, scores[top_idx]

def dense_retrieve_chunks(query, choices=None):
    idx, _ = dense_retrieve(query, chunk_embeddings, k=TOP_K)
    return [chunks[int(i)] for i in idx]  # cast to int
```

---

### 2. `IndexError: list index out of range` After Chunk Rebuild
**Problem:** After switching from character-based chunking to `chunk_by_markdown()`, the number of chunks changed (e.g., from 1,053 to a different count), but `chunk_embeddings` was still the old embedding matrix from the previous run. This caused `chunks[i]` to go out of bounds.

**Fix:** Always re-run embedding and BM25 indexing immediately after any chunk rebuild.

```python
# Must re-run after ANY change to chunks
chunk_embeddings = embed_model.encode(chunk_texts, ...)
tokenized_chunks = [word_tokenize(c["text"], ...) for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)
```

---

### 3. Score 0.06 Despite Pipeline Running — LLM Receiving Dict Objects
**Problem:** The original `build_prompt()` passed raw chunk dicts `{text, source}` as context strings. Python's string interpolation rendered them as `{'text': '...', 'source': '...'}` instead of the actual text content, so the LLM was reading Python dict representations rather than Thai document text.

**Fix:** Explicitly extract `.text` and `.source` from each chunk dict in `build_prompt()`.

```python
def build_prompt(question, choices, contexts):
    ctx_text = "\n\n".join([
        f"[เอกสาร {i+1} | {c['source']}]\n{c['text'][:600]}"
        for i, c in enumerate(contexts)
    ])
```

---

### 4. API 400 Bad Request Error
**Problem:** The ThaiLLM API returned `400 Bad Request` and the code retried indefinitely with exponential backoff, causing a `KeyboardInterrupt` after several minutes.

**Fix:** Added an explicit check for status code 400 to return `None` immediately without retrying (since the error is in the request itself, not a transient network issue) and logged the response body for debugging.

```python
if resp.status_code == 400:
    print(f"  400 Bad Request body: {resp.text[:300]}")
    return None  # don't retry — prompt has an issue
```

---

### 5. Low Retrieval Accuracy (Pred=9 for Most Questions)
**Problem:** After fixing the dict bug, many questions still returned answer 9 ("no data found"). The retrieval was not finding the right chunks because questions alone lacked the specific keywords present in the knowledge base.

**Fix:** Implemented **query expansion** — appending the first 4 content-specific choices (1–8) to the query string before retrieval. This significantly improved BM25 keyword matching for product names, specs, and policy terms.

```python
def build_query(question, choices):
    relevant_choices = [v for k, v in choices.items() if int(k) <= 8]
    choices_str = " ".join(relevant_choices[:4])
    return f"{question} {choices_str}"
```

---

### 6. Chunking Cuts Documents Mid-Sentence
**Problem:** The original fixed-size character chunking (`CHUNK_SIZE=512, OVERLAP=128`) split Thai documents at arbitrary character positions, often breaking mid-word, mid-sentence, or mid-table row. This destroyed the semantic coherence of chunks.

**Fix:** Switched to **Markdown-aware chunking** that splits on header boundaries (`## ...`, `### ...`) first, then falls back to size-based sub-splitting only for oversized sections. Also added a metadata prefix (`[หมวด: products | ไฟล์: ...]`) to every chunk so the LLM always knows the document context.

```python
def chunk_by_markdown(text, source, max_size=800):
    sections = re.split(r'\n(?=#+\s)', text)
    for section in sections:
        meta = f"[หมวด: {folder} | ไฟล์: {filename}]\n"
        ...
```

---

## 📁 Repository Structure

```
├── starter_kit_FahMai_RAG.ipynb   # Main notebook
├── submission.csv                  # Final predictions (100 rows)
└── README.md                       # This file
```

---

## 🔑 Key Lessons Learned

1. **Debug before scaling** — Always test 1 question end-to-end and print the raw prompt before running all 100.
2. **Re-embed after every chunk change** — Embeddings and chunk list must always be in sync.
3. **Chunking strategy matters more than model choice** — Switching from character-split to Markdown-aware chunking improved score by ~0.17 points.
4. **Query expansion is free retrieval improvement** — Adding choices to the retrieval query costs nothing and meaningfully boosts BM25 recall.
5. **Explicit `int()` casting for NumPy indices** — `chunks[np.int64(i)]` and `chunks[int(i)]` behave differently; always cast.

---

## 🙏 Acknowledgements

- [Super AI Engineer Program](https://superai.aiat.or.th/) for organizing the competition
- **Nutchanon Yongsatianchot, Ph.D.** for providing the Starter Kit that served as the foundation of this project
- **[Claude (Anthropic)](https://claude.ai)** — used as an AI pair-programmer throughout the 24-hour build. Every bug fix, architecture change, and prompt improvement in this project was iterated with Claude's assistance, from diagnosing the 562,059-character prompt bug to implementing query expansion and majority voting. Reaching a public score of 0.71 within 24 hours would not have been possible without it.
- [ThaiLLM](https://playground.thaillm.or.th/) for providing access to Thai language models
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for the multilingual embedding model
