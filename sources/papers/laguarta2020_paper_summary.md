# 📄 Paper Summary: COVID-19 Artificial Intelligence Diagnosis Using Only Cough Recordings

- **Authors:** Jordi Laguarta, Ferran Hueto, Brian Subirana
- **Affiliation:** MIT Auto-ID Laboratory (Cambridge, MA, USA) & Harvard University
- **Journal:** *IEEE Open Journal of Engineering in Medicine and Biology (OJEMB)*, Vol. 1, pp. 275–281, September 2020
- **DOI:** [10.1109/OJEMB.2020.3026928](https://doi.org/10.1109/OJEMB.2020.3026928)
- **Indexing:** PubMed PMID: 34812418 | PMCID: PMC8900604

---

## 1. Executive Summary & Core Hypothesis

The MIT team hypothesized that SARS-CoV-2 infection produces subtle, quantifiable neuromuscular and anatomical changes in respiratory dynamics that manifest in forced cough recordings—even in **completely asymptomatic individuals**. While imperceptible to the unaided human ear, these patterns can be classified using deep learning trained on acoustic biomarkers.

---

## 2. Dataset Collection & Demographics

- **Platform:** Web application (`opensigma.mit.edu`) launched between April and May 2020.
- **Total Participants:** 5,320 subjects.
  - 2,660 COVID-19 positive individuals (verified with official RT-PCR or clinical tests).
  - 2,660 COVID-19 negative control subjects.
- **Asymptomatic Cohort:** 48% of the COVID-19 positive participants self-reported zero physical symptoms at the time of recording.
- **Split:** 4,256 subjects (80%) for training; 1,064 subjects (20%) held out for blinded validation.

---

## 3. Four Acoustic Biomarkers Identified

The MIT architecture decomposes cough audio into four bio-acoustic dimensions:

1. **Vocal Cord Biomarker:**
   - Evaluates mucosal tension and fundamental frequency distortion caused by upper airway inflammation during the initial explosive cough compression.
2. **Sentiment / Neuromuscular Degradation Biomarker:**
   - Adapts acoustic classifiers originally researched at MIT for Alzheimer's disease to measure subtle central and peripheral motor control decline.
3. **Respiration / Lung Capacity Biomarker:**
   - Quantifies the total expiratory volume, peak flow acceleration, and duration of the turbulent cough decay phase.
4. **Cough Strength / Stridor Biomarker:**
   - Isolates abnormal harmonic energy concentrations in the **1.5 kHz to 4.0 kHz** band characteristic of subglottic airway narrowing.

---

## 4. Artificial Intelligence Architecture

- **Audio Preprocessing:** 64-band Log-Mel / MFCC spectrogram extraction.
- **Neural Network:**
  - Parallel feature extraction branches incorporating pre-trained **ResNet50** backbones.
  - Transfer learning applied from large-scale speech and respiratory datasets.
  - A Poisson acoustic biomarker integration layer producing a normalized scalar risk index (0.0 to 1.0).

---

## 5. Key Published Results

| Patient Category | Sensitivity (Recall) | Specificity | AUC |
| :--- | :--- | :--- | :--- |
| **All Official PCR Tested Subjects** | **98.5%** | **94.2%** | **0.97** |
| **Asymptomatic Only Subjects** | **100.0%** | **83.2%** | **0.94** |

---

## 6. How Our FHT Project Implements This Concept

Our project takes the MIT paper's architectural methodology and builds an accessible, working IoT physical prototype:
1. **Physical Sensing:** Replaces smartphone app upload with an **INMP441 MEMS microphone** on an **ESP32**, capturing 16 kHz 16-bit uncompressed PCM.
2. **Edge Computing:** Uses an adaptive RMS+MAD VAD state machine and a 2D CNN (3–4 convolutional blocks) running on a local PC server.
3. **Instant Triage:** Real-time WebSocket visual dashboard for instant kiosk/dormitory prescreening.
4. **Public Data Grounding:** Trained on validated open clinical datasets (**IISc Coswara** and **EPFL COUGHVID**) stored locally in `sources/`.
