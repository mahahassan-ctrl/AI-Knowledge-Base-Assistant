# AI Knowledge Base Assistant — Retrieval-Augmented Generation (RAG)

A document-grounded AI assistant that retrieves information from internal PDF documents, generates answers using a large language model (LLM), and provides source-page citations to make its responses easier to verify. The system also includes a fallback for questions that cannot be answered from the available knowledge base.

**Repository:** [AI-Knowledge-Base-Assistant](https://github.com/mahahassan-ctrl/AI-Knowledge-Base-Assistant)

## 1. Project Overview

Large language models can generate fluent responses that are incomplete, inaccurate, or unsupported by a particular organization's documents. A RAG system addresses this limitation by retrieving relevant information from a defined knowledge base before generating an answer.

This project implements an end-to-end RAG pipeline with:

- PDF ingestion and text extraction.
- Three text-chunking strategies.
- Two transformer-based embedding models.
- FAISS vector search.
- Top-k document retrieval.
- LLM-based answer generation using Gemini.
- Source-page citations and an unsupported-query fallback.
- A 40-question retrieval benchmark for comparing chunking and embedding configurations.
- Separate answer-quality evaluation tooling and manually reviewed outputs.

The implementation focuses on evaluating retrieval quality independently from answer-generation quality, since retrieving relevant information does not automatically guarantee a correct final answer.

## 2. System Architecture

The pipeline follows these stages:

1. **Document ingestion:** Load PDF files and extract their text.
2. **Chunking:** Divide long documents into smaller passages while preserving source metadata.
3. **Embedding generation:** Convert each passage into a dense numerical vector.
4. **Vector indexing:** Store the vectors in FAISS for efficient similarity search.
5. **Question retrieval:** Embed a user's question and retrieve the most relevant passages.
6. **Answer generation:** Provide the retrieved context to Gemini with instructions to answer from the available evidence.
7. **Citation and fallback:** Identify source-document pages and return a predefined refusal when sufficient supporting information is unavailable.
8. **Evaluation:** Measure retrieval performance and separately review generated answers and refusal behavior.

### Architecture diagram

```text
PDF knowledge base
        |
        v
Text extraction and metadata
        |
        v
Chunking strategies
(recursive / fixed / structure-based)
        |
        v
Embedding models
(MiniLM / MPNet)
        |
        v
FAISS vector indexes
        |
        v
User question
        |
        v
Question embedding and top-k retrieval
        |
        v
Retrieved passages + source metadata
        |
        v
Gemini answer generation
        |
        v
Grounded answer with citations
or unsupported-query fallback
```

The retrieval benchmark evaluates the retrieval portion of this pipeline. Answer correctness, citation quality, and refusal behavior are separate evaluation concerns.

## 3. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Pipeline implementation and evaluation |
| Embeddings | Sentence Transformers | Convert text into dense vectors |
| Embedding models | MiniLM and MPNet | Compare semantic retrieval configurations |
| Vector search | FAISS | Similarity search over document embeddings |
| LLM | Gemini 2.5 Flash | Generate answers from retrieved context |
| LLM SDK | Google Gen AI SDK | Communicate with Gemini |
| Application interface | Streamlit | Provide an interactive assistant |
| Evaluation | Python, JSON, CSV | Store benchmark questions, outputs, and results |

## 4. Document Ingestion and Chunking

### 4.1 PDF ingestion

The source knowledge base contains internal-style employee, HR, IT, and security policies in PDF format. The ingestion pipeline extracts the document text and retains source information required for citation and evaluation.

Source metadata is important because a response should identify not only the information retrieved but also the document and page from which that information originated.

### 4.2 Why chunking matters

Embedding an entire long document as one vector can make precise retrieval difficult. Chunking divides the document into smaller passages so the retrieval system can identify the portions most relevant to a question.

The implementation compares three strategies:

**Recursive chunking**

Splits text using a hierarchy of separators, attempting to preserve natural text boundaries before splitting into smaller passages. This can retain more coherent context than splitting at arbitrary character positions.

**Fixed-size chunking**

Divides text into passages of a specified size, with overlapping text between consecutive chunks. It provides predictable chunk lengths but can split sentences or concepts across boundaries.

**Structure-based chunking**

Uses document structure, such as headings and paragraph boundaries, to create passages that follow the organization of the source material.

The configured chunk size is 500 characters, with 100 characters of overlap where applicable.

### 4.3 Chunk overlap

Overlap helps preserve information that crosses a chunk boundary. Without overlap, a relevant statement may be divided between two passages, making it harder to retrieve either passage independently.

For chunk size \(C\) and overlap \(O\), the approximate step between consecutive chunks is:

\[
S=C-O
\]

With the configured values:

\[
S=500-100=400 \text{ characters}
\]

This means successive chunks begin approximately 400 characters apart when the strategy uses the configured overlap.

Chunk size and overlap are hyperparameters: they should be selected based on retrieval performance rather than assumed to be optimal.

## 5. Embedding Models and FAISS

### 5.1 Sentence embeddings

An embedding model maps a passage of text to a dense vector:

\[
f:\text{text}\rightarrow\mathbb{R}^{d}
\]

Semantically similar passages can have vectors that are close in the embedding space, even when they do not share exactly the same words.

The project compares:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

MiniLM provides a compact embedding model, while MPNet provides an alternative semantic representation. Their actual retrieval performance is measured empirically in the benchmark.

### 5.2 FAISS similarity search

The vectors are indexed in FAISS. At query time, the system retrieves the top \(k\) passages most similar to the question representation.

For normalized vectors, cosine similarity is:

\[
\operatorname{cos}(q,d)=
\frac{q\cdot d}{\|q\|\|d\|}
\]

where \(q\) is the question vector and \(d\) is a document vector.

For normalized vectors, cosine similarity is equivalent to the inner product. The exact metric used by an index depends on its configuration.

The current retrieval configuration uses \(k=3\), meaning the three highest-ranked retrieved chunks are supplied to the answer-generation stage.

Retrieval ranking identifies likely relevant evidence; it does not prove that the generated answer is correct.

## 6. LLM Answer Generation and Grounding

The answer-generation stage uses Gemini 2.5 Flash through the Google Gen AI SDK.

The retrieved passages are provided as context, alongside instructions to:

- Answer using the available document evidence.
- Avoid inventing facts that are not supported by the context.
- Include source-document and page information where available.
- Return a predefined fallback when sufficient evidence is unavailable.

This is the grounding component of the system: the model is expected to base its answer on retrieved source material rather than relying solely on its general pretrained knowledge.

### Unsupported-query fallback

The assistant defines a standard refusal message:

> I couldn't find enough information in the available documents to answer that question reliably.

The fallback is designed for questions that cannot be answered reliably from the available knowledge base.

For example, a question about a company's current stock price should not receive an invented numerical answer when the knowledge base contains only internal policy documents.

The system's fallback and prompt instructions reduce the risk of unsupported responses. However, an instruction to refuse does not guarantee that an LLM will always refuse correctly. That behavior requires testing against relevant and irrelevant contexts, including cases where retrieved passages appear similar to a question but do not support its answer.

### Page-level citations

Source citations help users verify the answer against the original documents. A citation should identify the correct document and page, and the cited page should contain evidence supporting the associated claim.

Citation quality is distinct from citation formatting: a syntactically valid citation can still point to an irrelevant or incorrect page.

## 7. Retrieval Evaluation

### 7.1 Evaluation dataset

A dedicated benchmark of 40 answerable questions was created in:

`evaluation/retrieval_eval_questions.json`

Each question contains expected relevant source-page references. These references act as the ground truth against which the retrieved results are evaluated.

The benchmark compares:

- Three chunking strategies.
- Two embedding models.
- Top-3 retrieval.

This gives six chunking-and-embedding configurations, evaluated on the same set of questions.

### 7.2 Recall@k

Recall@k measures the proportion of evaluation questions for which relevant source material appears within the top \(k\) retrieved results.

For a page-level relevance criterion:

\[
\operatorname{Recall@k}=
\frac{\sum_{i=1}^{N}
\mathbf{1}[\text{relevant source found in top }k]}
{N}\times100\%
\]

Here:

- \(N\) is the number of evaluation questions.
- \(\mathbf{1}[\cdot]\) equals 1 when the relevance condition is satisfied and 0 otherwise.
- \(k\) is the number of retrieved results considered.

The benchmark uses \(k=3\). A question counts as a hit when a retrieved chunk contains a relevant source page specified in the question's ground-truth references.

For 40 questions, each question contributes 2.5 percentage points to the result:

\[
\frac{1}{40}\times100=2.5\%
\]

A result of 36 successful retrievals therefore gives:

\[
\operatorname{Recall@3}=
\frac{36}{40}\times100=90\%
\]

Recall@3 measures retrieval relevance, not answer correctness.

### 7.3 Measured benchmark results

| Chunking strategy | Embedding model | Relevant retrievals | Recall@3 |
|---|---|---:|---:|
| Recursive | MiniLM | 36/40 | 90.0% |
| Recursive | MPNet | 36/40 | 90.0% |
| Fixed-size | MiniLM | 35/40 | 87.5% |
| Fixed-size | MPNet | 30/40 | 75.0% |
| Structure-based | MiniLM | 36/40 | 90.0% |
| Structure-based | MPNet | 36/40 | 90.0% |

**Best measured Recall@3: 90.0%**, achieved by recursive and structure-based chunking with either tested embedding model.

The results show that retrieval performance depends on the combination of chunking strategy and embedding model. In this benchmark, fixed-size chunking with MPNet produced the lowest result.

The dataset contains 40 questions, so the results are useful for comparing these configurations on the selected benchmark, but should not be interpreted as a guarantee of performance on arbitrary documents or user questions.

### 7.4 Additional retrieval metrics

The same retrieval outputs can be used to calculate additional metrics without making any new LLM API requests, provided the ranked results and relevance labels are available.

**Recall@1**

Measures whether a relevant source appears in the first result:

\[
\operatorname{Recall@1}=
\frac{\text{Questions with a relevant first result}}{N}
\]

**Recall@5**

Measures whether a relevant source appears within the first five results:

\[
\operatorname{Recall@5}=
\frac{\text{Questions with a relevant result in top 5}}{N}
\]

These metrics are only meaningful if the evaluator actually retrieves and records five results.

**Mean Reciprocal Rank (MRR)**

MRR measures how highly the first relevant result is ranked:

\[
\operatorname{MRR}=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}
\]

where \(r_i\) is the rank of the first relevant result for question \(i\). If no relevant result is found, its reciprocal rank is zero.

Recall measures whether relevant evidence was retrieved within a cutoff; MRR additionally rewards systems that place the first relevant result nearer the top.

## 8. Answer-Quality and Refusal Evaluation

Retrieval evaluation is not sufficient to establish the quality of the final generated answer. The project therefore includes a separate answer-quality evaluation script.

### 8.1 Answerable questions

The answer-generation test set uses the 40 questions in the retrieval benchmark. The intended evaluation compares generated answers with the expected information in the source documents and checks their supporting citations.

An answer should be marked correct only when it provides the expected information without material factual errors and its claims are adequately supported by the available sources.

### 8.2 Grounded answer accuracy

Grounded answer accuracy is calculated as:

\[
\text{Grounded Accuracy}=
\frac{\text{Correct, adequately supported answers}}
{\text{Successfully evaluated answerable questions}}
\times100\%
\]

For example, if 36 of 40 generated answers pass the predefined correctness and support criteria, the result is 90%.

The denominator must contain questions with evaluable generated answers. API errors must not be counted as incorrect answers or removed silently without reporting the resulting sample size.

### 8.3 Unsupported questions and refusal correctness

The evaluation script includes 20 unsupported questions covering information such as company revenue, executive compensation, customer numbers, stock prices, market valuation, and future forecasts.

These questions are designed to test whether the assistant avoids answering beyond the evidence in the knowledge base.

Refusal correctness is:

\[
\text{Refusal Correctness}=
\frac{\text{Unsupported questions correctly refused}}
{\text{Successfully evaluated unsupported questions}}
\times100\%
\]

For example, 19 correct refusals out of 20 successfully evaluated unsupported questions would produce 95%.

A correct refusal must avoid presenting unsupported information as established fact. An API error is not a refusal, and a question should not count as a correct refusal merely because the response is empty.

### 8.4 Manual review and fallback testing

The answer-generation run encountered Gemini API quota limitations, leaving only six successfully generated answers in the recorded output. Those saved answers can be manually reviewed against the source PDFs without additional API calls.

The fallback behavior can also be tested locally by supplying empty retrieval results or mocked irrelevant documents and checking that the application returns the configured refusal message.

These methods provide useful evidence about the implementation, but they answer different questions:

- Manual review checks the correctness of the particular answers that were generated.
- Mocked unit tests check the fallback code path.
- A full answer-quality benchmark measures how the live model performs across the entire evaluation set.

For reproducibility, grounded answer accuracy and refusal correctness should be reported only after the relevant answers have been generated, scored against explicit criteria, and counted with the correct denominators.

## 9. Retrieve-and-Rerank: Evaluation Considerations

A retrieve-and-rerank architecture can use a fast first-stage retriever to select candidate passages and a second-stage ranking model to reorder those candidates by relevance.

The general pipeline is:

1. Retrieve candidate chunks from FAISS.
2. Score the candidates using a reranker.
3. Reorder the candidates by reranker score.
4. Pass the highest-ranked passages to the LLM.

A reranker may improve ranking quality, but this must be measured rather than assumed.

To evaluate its effect, use the same questions, source-page ground truth, candidate set, and evaluation rules for both the baseline retriever and the reranked pipeline.

Absolute improvement is:

\[
\Delta_{\text{pp}}=
\operatorname{Recall@3}_{\text{new}}
-\operatorname{Recall@3}_{\text{baseline}}
\]

where the difference is expressed in percentage points.

Relative improvement is:

\[
\text{Relative Improvement}=
\frac{\text{New}-\text{Baseline}}{\text{Baseline}}
\times100\%
\]

For example, an increase from 82.5% to 92.5% would be an improvement of 10 percentage points, or approximately 12.12% relative to the baseline. These figures are an illustration of the formulas, not the measured results of the current benchmark.

A reranking improvement should only be reported after running and saving the corresponding baseline and reranked experiments.

## 10. Project Structure

```text
AI-Knowledge-Base-Assistant/
├── app.py
├── chatbot.py
├── chunking.py
├── embeddings.py
├── retrieval.py
├── evaluate_retrieval.py
├── evaluate_answer.py
├── data/
│   └── PDF knowledge sources
├── evaluation/
│   ├── retrieval_eval_questions.json
│   └── results/
│       ├── benchmark_summary.csv
│       ├── benchmark_details.json
│       ├── answer_quality_outputs.json
│       └── answer_quality_review.csv
├── vectorstore/
│   ├── minilm/
│   └── mpnet/
├── requirements.txt
└── README.md
```

The vector-store folders contain indexes generated for the embedding configurations. They should be rebuilt if the underlying documents, chunking configuration, or embedding model changes.

## 11. Setup and Execution

### Prerequisites

- Python environment compatible with the packages in `requirements.txt`.
- The project PDF files.
- A Gemini API key for live answer generation.

### Install dependencies

```bash
git clone https://github.com/mahahassan-ctrl/AI-Knowledge-Base-Assistant.git
cd AI-Knowledge-Base-Assistant
pip install -r requirements.txt
```

### Configure the API key

Set `GEMINI_API_KEY` as an environment variable or a Codespaces secret. Do not place API keys directly in source code or commit them to Git.

For a local Linux shell, the key can be exported for the current terminal session:

```bash
export GEMINI_API_KEY="your_api_key"
```

Use your actual key locally; never commit it to the repository.

### Run the retrieval benchmark

```bash
python evaluate_retrieval.py
```

The retrieval benchmark uses the local embedding models and FAISS indexes and does not require a Gemini generation request.

### Run the answer-quality evaluation

```bash
python evaluate_answer.py
```

This script invokes Gemini and is subject to API rate limits and usage quotas. Check the existing evaluation output before rerunning it, and avoid restarting a complete evaluation unnecessarily.

### Launch the Streamlit application

```bash
streamlit run app.py
```

If the application requires prebuilt indexes, ensure that the appropriate vector stores have been generated before launching it.

## 12. Reproducibility and Limitations

The project stores evaluation questions and result files so retrieval experiments can be inspected and compared.

For reliable comparisons:

- Keep the same evaluation questions and relevance labels across configurations.
- Keep the retrieval cutoff consistent when comparing Recall@k.
- Record the chunking settings and embedding model used for each run.
- Separate retrieval failures from answer-generation failures.
- Review citations against the original PDFs.
- Report the sample size and denominator alongside percentages.
- Avoid treating API errors as incorrect answers or successful refusals.
- Do not compare metrics produced with different evaluation rules as if they were directly equivalent.

The current retrieval benchmark covers 40 questions and a limited set of PDF documents. Its results do not guarantee equivalent performance on larger or more diverse knowledge bases.

Answer-generation evaluation is additionally limited by the number of successful recorded generations. The manually reviewed responses and fallback tests provide partial evidence, but they do not establish full-dataset grounded accuracy or live-model refusal correctness.

## Hallucination Mitigation and Grounded Answer Generation

A key design goal of this project was to reduce hallucinations by ensuring that answers are based on the retrieved policy documents rather than the language model's general knowledge.

### 1. Retrieval-grounded context

The system retrieves the top 3 relevant chunks from the FAISS vector store for each user query. These chunks provide the evidence used to generate an answer.

The retrieval process follows these steps:

1. Convert the user's question into an embedding using the selected sentence-transformer model.
2. Search the FAISS index for semantically similar document chunks.
3. Retrieve the top-\(k\) chunks, where \(k=3\).
4. Pass the retrieved context to the LLM for answer generation.

Restricting the answer-generation context to retrieved documents helps reduce unsupported claims. However, retrieval alone cannot guarantee factual correctness, particularly when the relevant information is missing from the index or the retrieved chunks are incomplete.

### 2. Document-grounded prompting

The LLM is instructed to answer using the supplied document context rather than inventing missing information. When the context does not contain sufficient evidence to answer a question reliably, the system should decline to provide a factual answer.

This approach establishes a clear distinction between information supported by the knowledge base and information that cannot be verified from the available documents.

### 3. Unsupported-query fallback

The application defines the following fallback message:

> "I couldn't find enough information in the available documents to answer that question reliably."

This fallback is intended for questions that cannot be answered adequately from the retrieved evidence. For example, if a user asks for company revenue or a stock price that is not documented in the indexed policies, the system should avoid inventing a value and instead communicate that the information is unavailable.

The fallback mechanism helps prevent the model from filling gaps in the knowledge base with plausible but unsupported answers.

**Important limitation:** A prompt-based fallback depends on the model following its instructions. It is not, by itself, a guarantee that every unsupported query will be refused correctly.

### 4. Page-level source citations

Answers include source references that identify the document and relevant page, allowing users to trace generated claims back to the original material.

Citations improve transparency and make it easier to manually verify answers. They also help reviewers identify cases where an answer is unsupported by its cited source.

A citation should not automatically be treated as proof of correctness: the cited passage must actually support the associated claim.

### 5. Evaluation and manual verification

The project uses a 40-question retrieval evaluation set to measure whether the correct source pages appear among the top 3 retrieved results. The best recorded configuration achieved **90% Recall@3 (36 out of 40 questions)**.

This metric evaluates retrieval coverage, not the factual accuracy of generated answers. Answer quality and refusal behavior therefore require separate evaluation.

Generated answers can be manually reviewed to check:

- **Groundedness:** Are the factual claims supported by the retrieved documents?
- **Citation validity:** Do the cited document and page correspond to the supporting evidence?
- **Unsupported-query handling:** Does the system decline to answer when the available context is insufficient?
- **Completeness:** Does the answer omit important qualifications or conditions present in the source?

Manual review is useful for identifying failure cases, but answer-accuracy and refusal percentages should only be reported after the relevant test cases have been scored.

### 6. Remaining limitations

The implemented measures reduce hallucination risk but do not eliminate it. Potential failure cases include incorrect retrieval, incomplete context, misleading citations, and the model generating claims that go beyond the supplied evidence.

Further improvements could include claim-level citation verification, automated checks that every answer is supported by retrieved passages, confidence thresholds for refusing uncertain answers, and a larger manually scored evaluation set.

Overall, the project uses retrieval grounding, constrained prompting, page-level citations, and an unsupported-query fallback as complementary measures for improving the reliability and auditability of generated answers.
## Author

**Maha Hassan**  
BSc Artificial Intelligence, Deggendorf Institute of Technology

GitHub: [mahahassan-ctrl](https://github.com/mahahassan-ctrl)