# AI-Based Intelligent Resume Screening and Job Recommendation System

A beginner-friendly B.Tech mini project that extracts information from resumes, compares resumes with real job records, and recommends relevant jobs through Streamlit.

## Project Description

The system accepts a PDF, DOCX, or TXT resume and compares it with the public Hugging Face dataset `batuhanmtl/job-skill-set`.

It provides:

- Resume text extraction
- Skill extraction
- Education and experience extraction when present
- Project and certification section detection
- Resume-to-job skill comparison
- TF-IDF resume-to-job-description similarity
- Transparent matching scores
- Ranked job recommendations
- Matched and missing skill explanations

The system does not use names, IDs, or sensitive personal characteristics for matching.

## Dataset Description

### Resume dataset

Source file: `resume dataset.zip`

The ZIP contains `ml_resume_dataset_4500.csv` with 4,500 rows and 9 original columns:

- `id`
- `name`
- `years_experience`
- `highest_degree`
- `skills`
- `current_title`
- `has_portfolio`
- `raw_text`
- `label`

The original ZIP is read without extraction and is never overwritten.

### Job dataset

Source: [batuhanmtl/job-skill-set](https://huggingface.co/datasets/batuhanmtl/job-skill-set)

The dataset contains 1,167 job records and these five fields:

- `job_id`
- `category`
- `job_title`
- `job_description`
- `job_skill_set`

There are five categories and 723 distinct job titles. The job dataset has no separate education, experience, salary, company, or location columns. Education and experience are considered only when explicit information is found in `job_description`.

### Supplemental job records

The local file `data/additional_jobs.csv` adds 17 meaningful technology roles without changing the Hugging Face dataset. It uses the same five columns and unique IDs in the `4000000001` to `4000000017` range. The combined catalog contains 1,184 jobs when the Hugging Face source is available.

The added roles are Machine Learning Engineer, AI Engineer, Data Scientist, Data Analyst, Data Engineer, NLP Engineer, Computer Vision Engineer, Generative AI Engineer, Python Developer, Software Engineer, Backend Developer, Full Stack Developer, Cloud Engineer, DevOps Engineer, Cybersecurity Analyst, Database Engineer, and SQL Developer.

## Methodology

1. Load and validate both datasets.
2. Clean whitespace and preserve meaningful technical text.
3. Parse resume comma-separated skills.
4. Parse the job dataset's string-formatted `job_skill_set` lists.
5. Extract resume skills, education level, experience years, projects, and certifications where available.
6. Compare normalized resume skills with normalized job skills.
7. Calculate TF-IDF cosine similarity between resume text and job description.
8. Extract explicit education and experience requirements from job descriptions when possible.
9. Rank jobs by the transparent matching score.
10. Display the result in Streamlit.

## Architecture

```text
Resume upload
    -> resume_parser.py
    -> skill_extractor.py
    -> candidate profile

Hugging Face job dataset
    -> data_loader.py
    -> data_preprocessor.py
    -> validated job records

Candidate profile + job records
    -> matcher.py
    -> recommender.py
    -> Streamlit results
```

## Folder Structure

```text
resume_screening_system/
├── app.py
├── demo.py
├── README.md
├── requirements.txt
├── resume dataset.zip
├── data/
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── evaluation.py
│   ├── matcher.py
│   ├── recommender.py
│   ├── resume_parser.py
│   └── skill_extractor.py
└── tests/
    ├── README.md
    ├── test_data_preprocessing.py
    ├── test_evaluation.py
    ├── test_matcher.py
    ├── test_phase10.py
    ├── test_recommender.py
    ├── test_resume_parser.py
    └── test_skill_extractor.py
```

## Technologies

- Python
- pandas for tabular data
- requests for Hugging Face dataset access
- scikit-learn for TF-IDF and cosine similarity
- pdfplumber for PDF text extraction
- python-docx for DOCX text extraction
- Streamlit for the user interface
- pytest for testing

## Matching Algorithm

### Skill comparison

Resume skills and `job_skill_set` are normalized to lowercase sets.

```text
matched_skills = resume_skills intersection job_skills
missing_skills = job_skills minus resume_skills
```

Skill overlap is calculated as:

```text
matched job skills / total listed job skills
```

### Text comparison

The resume text and `job_description` are converted into TF-IDF vectors. Cosine similarity measures their textual similarity.

### Education and experience

The resume dataset supplies `highest_degree` and `years_experience`. The job dataset does not supply equivalent columns, so the system searches `job_description` for explicit degree and experience requirements.

If a requirement is absent, the result is shown as `unavailable`. It is not treated as a failed requirement.

### Current score parameters

The implementation uses provisional, normalized weights:

- Text similarity: 40%
- Skill overlap: 40%
- Education fit: 10% when available
- Experience fit: 10% when available

These weights are not claimed to be optimal. They should be evaluated with manually reviewed resume-job relevance judgments before being finalized.

## Evaluation

No classifier is trained because the available datasets do not provide reliable recommendation ground truth. The resume `label` field has no documented meaning and is not used as a recommendation target.

Implemented checks include:

- Missing values
- Duplicate rows
- Duplicate IDs
- Skill-list parsing
- Score range validation
- Ranking order validation
- Precision@K and Recall@K when manually reviewed relevant job IDs are supplied

## Running the Project

From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The Streamlit application is available at `http://localhost:8501` by default.

## Results

The current automated test suite contains 22 passing tests. The verified source datasets contain:

- 4,500 resume records
- 1,167 job records
- No missing values in the original inspected datasets
- No exact duplicate rows in the original inspected datasets
- Unique resume IDs and job IDs

These are dataset and test results, not claims that the recommender has a measured real-world accuracy. Recommendation accuracy requires manually reviewed relevance labels or historical application outcomes.

## Limitations

- The resume skill vocabulary is limited and rule-based.
- Scanned image PDFs may not contain extractable text.
- TF-IDF may miss synonyms and deeper semantic meaning.
- Job education and experience requirements are embedded in descriptions rather than structured columns.
- The job dataset does not contain company, location, salary, or employment type.
- The current score weights are provisional.
- The resume dataset's `label` meaning is undocumented.

## Future Scope

- Add OCR support for scanned resumes.
- Expand skill normalization using a documented skills taxonomy.
- Compare TF-IDF with sentence embeddings.
- Add manually reviewed relevance labels for evaluation.
- Extract structured requirements from job descriptions more accurately.
- Add filters based on fields that become available in a future job dataset.
- Add fairness and bias auditing before real-world use.

## References

- [Hugging Face Job Skill Set Dataset](https://huggingface.co/datasets/batuhanmtl/job-skill-set)
- [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn cosine similarity documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [Streamlit documentation](https://docs.streamlit.io/)
