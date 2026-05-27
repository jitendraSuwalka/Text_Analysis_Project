# NLP Text Classification & Sentiment Analysis Pipeline

End-to-end Natural Language Processing and Machine Learning pipeline for multi-class text classification, sentiment analysis, and topic discovery using the 20 Newsgroups benchmark dataset.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | **88.5%** | **89.1%** | **88.5%** | **88.3%** |
| Logistic Regression | 87.8% | 88.2% | 87.8% | 87.7% |
| Linear SVM | 88.1% | 88.5% | 88.1% | 88.0% |

## Pipeline Overview

```
20 Newsgroups Dataset (4,864 documents, 5 categories)
    │
    ├── 1. Text Preprocessing (lowercasing, stopword removal, special char removal)
    │
    ├── 2. Exploratory Data Analysis (category distribution, word counts, word clouds)
    │
    ├── 3. Sentiment Analysis (TextBlob polarity & subjectivity scoring)
    │
    ├── 4. TF-IDF Vectorisation (feature extraction, top terms per category)
    │
    ├── 5. Topic Modelling (Latent Dirichlet Allocation — 5 discovered topics)
    │
    ├── 6. Classification (Naive Bayes, Logistic Regression, Linear SVM)
    │
    └── 7. Evaluation (accuracy, precision, recall, F1, confusion matrices)
```

## Categories Analysed

- `rec.sport.baseball` — Sports
- `sci.med` — Medical Science
- `comp.graphics` — Computer Graphics
- `talk.politics.guns` — Politics
- `soc.religion.christian` — Religion

## Visualisations (16 plots generated)

The pipeline produces 16 publication-quality visualisations saved to `text_analysis_plots/`:

| # | Plot | Purpose |
|---|------|---------|
| 01 | Category distribution | Class balance check |
| 02 | Word count boxplot | Document length by category |
| 03 | Word count histogram | Overall length distribution |
| 04 | Top 20 words | Most frequent terms across corpus |
| 05 | Word clouds by category | Visual term frequency per class |
| 06 | Sentiment distribution | Positive/neutral/negative split |
| 07 | Polarity vs subjectivity | Sentiment scatter analysis |
| 08 | Mean polarity by category | Category-level sentiment comparison |
| 09 | Sentiment stacked bars | Sentiment breakdown per category |
| 10 | TF-IDF top terms | Highest-weighted features per class |
| 11 | LDA topic heatmap | Topic-term weight distribution |
| 12 | Document-topic distribution | Topic assignment across documents |
| 13 | Model comparison | Accuracy bar chart (3 classifiers) |
| 14 | Confusion matrices | Per-model classification errors |
| 15 | Classification report heatmap | Precision/recall/F1 by class |
| 16 | Summary dashboard | Combined overview of all results |

## Tech Stack

- **Language:** Python 3.10+
- **NLP:** NLTK, TextBlob, WordCloud
- **ML:** Scikit-learn (MultinomialNB, LogisticRegression, LinearSVC)
- **Feature Extraction:** TF-IDF Vectoriser, CountVectoriser
- **Topic Modelling:** Latent Dirichlet Allocation (LDA)
- **Visualisation:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy

## Setup & Run

```bash
# Clone the repository
git clone https://github.com/jitendraSuwalka/Text_Analysis_Project.git
cd Text_Analysis_Project

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python text_analysis_project.py
```

All 16 visualisations will be saved to the `text_analysis_plots/` directory.

## Requirements

```
numpy
pandas
matplotlib
seaborn
nltk
textblob
scikit-learn
wordcloud
```

## Project Structure

```
Text_Analysis_Project/
├── text_analysis_project.py    # Complete NLP/ML pipeline (single script)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── text_analysis_plots/        # Generated visualisations (created on run)
    ├── 01_category_distribution.png
    ├── 02_word_count_boxplot.png
    ├── ...
    └── 16_summary_dashboard.png
```

## Key Findings

- **Best classifier:** Naive Bayes achieved the highest accuracy at 88.5%, outperforming Logistic Regression (87.8%) and Linear SVM (88.1%) on this multi-class text classification task.
- **Sentiment patterns:** Medical and religious texts showed higher subjectivity, while computer graphics documents were predominantly neutral.
- **Topic coherence:** LDA successfully discovered 5 distinct topics aligning closely with the 5 ground-truth categories.
- **Feature importance:** TF-IDF revealed strong category-specific vocabulary, with domain terms (e.g., "patient", "church", "image") appearing as top discriminators.

## Author

**Jitendra Suwalka**
MSc Data Science — University of Bristol
- [LinkedIn](https://www.linkedin.com/in/jitendra-suwalka-ds)
- [GitHub](https://github.com/jitendraSuwalka)
