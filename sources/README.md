# 📚 Data & Paper Sources Directory (`sources/`)

This directory houses all benchmark datasets, metadata, sample audio recordings, and foundational academic papers used in the **Cough Prescreening System**.

---

## Directory Organization

```
sources/
├── coswara/
│   ├── combined_data.csv          # Full metadata catalog (IISc Bangalore, 2,500+ participants)
│   ├── 20200413.csv               # Session-specific participant metadata
│   └── Extracted_data/
│       └── 20200413/              # 76 real participant directories (684 WAV files):
│           ├── <participant_id>/
│           │   ├── cough-heavy.wav
│           │   ├── cough-shallow.wav
│           │   ├── breathing-deep.wav
│           │   ├── breathing-shallow.wav
│           │   ├── vowel-a.wav
│           │   ├── vowel-e.wav
│           │   ├── vowel-o.wav
│           │   ├── counting-fast.wav
│           │   ├── counting-normal.wav
│           │   └── metadata.json
├── coughvid/
│   ├── sample_recordings/
│   │   ├── cough.wav              # Clinical cough sample from EPFL COUGHVID
│   │   └── not_cough.wav          # Ambient non-cough control sound
│   ├── models/
│   │   ├── cough_classifier       # Pre-trained XGBoost cough detection model
│   │   └── cough_classification_scaler
│   ├── src/                       # EPFL DSP feature extraction routines
│   ├── environment.yml
│   └── README.md
└── papers/
    ├── laguarta2020_full_article.xml    # Complete unabridged scientific XML (PMC8900604)
    ├── laguarta2020_paper_summary.md    # Formatted study summary & biomarker analysis
    └── citations.bib                    # BibTeX citations for MIT paper, Coswara & Coughvid
```

---

## 1. Coswara Dataset (Indian Institute of Science, Bangalore)
- **Institution:** IISc Bangalore (Department of Electrical Engineering).
- **Files Stored:**
  - `combined_data.csv`: Master metadata mapping participant IDs to COVID-19 status, symptoms, age, gender, and pre-existing medical conditions.
  - `Extracted_data/20200413/`: 684 individual audio recordings spanning 76 human participants with both cough and non-cough control respiratory sounds.
- **Link to Training Pipeline:**
  To train the PyTorch CNN on this extracted Coswara data, run:
  ```bash
  ./venv/bin/python -m server.train --coswara-dir sources/coswara/Extracted_data --epochs 15
  ```

---

## 2. COUGHVID Dataset (EPFL, Switzerland)
- **Institution:** École Polytechnique Fédérale de Lausanne (Embedded Systems Laboratory).
- **Files Stored:**
  - `sample_recordings/`: Verified pulmonologist-labeled cough recordings and control ambient audio.
  - `models/`: EPFL's pre-trained XGBoost cough detection model (`cough_classifier`) used for acoustic SNR validation.
  - `src/`: Audio DSP and feature computation algorithms.

---

## 3. Academic Paper Reference (MIT Auto-ID Lab)
- **Paper:** *"COVID-19 Artificial Intelligence Diagnosis Using Only Cough Recordings"*
- **Authors:** Jordi Laguarta, Ferran Hueto, Brian Subirana (*IEEE OJEMB 2020*)
- **Stored Artifacts:**
  - `laguarta2020_full_article.xml`: Complete NCBI PubMed Central full-text open-access archive (PMC8900604 / PMID: 34812418).
  - `laguarta2020_paper_summary.md`: Detailed synthesis of the four bio-acoustic markers and CNN architecture.
  - `citations.bib`: Official BibTeX citations ready to import into LaTeX or project report reference managers.
