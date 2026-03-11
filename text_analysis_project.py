"""
========================================================================
 TEXT ANALYSIS ML PROJECT
 -----------------------------------------------------------------------
 This project demonstrates a complete NLP/ML pipeline:
   1. Dataset loading (20 Newsgroups)
   2. Text preprocessing & cleaning
   3. Exploratory Data Analysis with rich visualizations
   4. Sentiment Analysis (TextBlob)
   5. TF-IDF Vectorization
   6. Topic Modeling (LDA)
   7. Text Classification (Naive Bayes, Logistic Regression, SVM)
   8. Model Comparison & Evaluation plots
========================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from collections import Counter

# NLP libraries
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.decomposition import LatentDirichletAllocation

# WordCloud
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ── Download required NLTK data ─────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Plotting style ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
sns.set_theme(style="whitegrid", palette="deep")

# ── Output directory ────────────────────────────────────────────────
OUTPUT_DIR = "text_analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 1. LOAD DATASET
# ====================================================================
print("=" * 65)
print(" STEP 1: Loading 20 Newsgroups Dataset")
print("=" * 65)

# Pick 5 diverse categories for clearer analysis
CATEGORIES = [
    'rec.sport.baseball',
    'sci.med',
    'comp.graphics',
    'talk.politics.guns',
    'soc.religion.christian',
]

newsgroups = fetch_20newsgroups(
    subset='all',
    categories=CATEGORIES,
    remove=('headers', 'footers', 'quotes'),
    random_state=42,
)

df = pd.DataFrame({
    'text': newsgroups.data,
    'target': newsgroups.target,
})
df['category'] = df['target'].map(lambda t: newsgroups.target_names[t])

print(f"  Total documents loaded : {len(df)}")
print(f"  Categories             : {CATEGORIES}")
print(f"  Shape                  : {df.shape}\n")

# ====================================================================
# 2. TEXT PREPROCESSING
# ====================================================================
print("=" * 65)
print(" STEP 2: Text Preprocessing")
print("=" * 65)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase, remove non-alpha, strip stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].astype(str).apply(clean_text)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
df['char_count'] = df['clean_text'].apply(lambda x: len(x))

print("  ✓ Cleaned text (lowercased, removed stopwords & special chars)")
print(f"  Average word count : {df['word_count'].mean():.1f}")
print(f"  Median word count  : {df['word_count'].median():.1f}\n")

# ====================================================================
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ====================================================================
print("=" * 65)
print(" STEP 3: Exploratory Data Analysis & Visualizations")
print("=" * 65)

# ── 3a. Document count per category ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
category_counts = df['category'].value_counts()
colors = sns.color_palette("viridis", len(category_counts))
bars = ax.barh(category_counts.index, category_counts.values, color=colors)
for bar, val in zip(bars, category_counts.values):
    ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontweight='bold', fontsize=11)
ax.set_xlabel("Number of Documents")
ax.set_title("Document Count per Category", fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_category_distribution.png")
plt.close()
print("  ✓ Saved 01_category_distribution.png")

# ── 3b. Word-count distribution by category ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='word_count', hue='category',
            palette='Set2', legend=False, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.set_title("Word Count Distribution by Category", fontweight='bold')
ax.set_xlabel("Category")
ax.set_ylabel("Word Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_word_count_boxplot.png")
plt.close()
print("  ✓ Saved 02_word_count_boxplot.png")

# ── 3c. Word-count histogram ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['word_count'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(df['word_count'].mean(), color='red', ls='--', lw=2,
           label=f"Mean = {df['word_count'].mean():.0f}")
ax.axvline(df['word_count'].median(), color='orange', ls='--', lw=2,
           label=f"Median = {df['word_count'].median():.0f}")
ax.legend(fontsize=11)
ax.set_title("Overall Word Count Distribution", fontweight='bold')
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_word_count_histogram.png")
plt.close()
print("  ✓ Saved 03_word_count_histogram.png")

# ── 3d. Top-20 most common words (bar chart) ────────────────────────
all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words).most_common(20)
words, counts = zip(*word_freq)

fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette("magma_r", len(words))
ax.barh(list(reversed(words)), list(reversed(counts)), color=palette)
ax.set_title("Top 20 Most Frequent Words", fontweight='bold')
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_top20_words.png")
plt.close()
print("  ✓ Saved 04_top20_words.png")

# ── 3e. Word Clouds per Category ────────────────────────────────────
n_cats = len(CATEGORIES)
fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5))
colormaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']

for i, cat in enumerate(sorted(df['category'].unique())):
    text = ' '.join(df[df['category'] == cat]['clean_text'])
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap=colormaps[i % len(colormaps)],
                   max_words=80).generate(text)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(cat.split('.')[-1].title(), fontweight='bold', fontsize=13)
    axes[i].axis('off')

fig.suptitle("Word Clouds by Category", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_wordclouds_by_category.png", bbox_inches='tight')
plt.close()
print("  ✓ Saved 05_wordclouds_by_category.png")

# ====================================================================
# 4. SENTIMENT ANALYSIS (TextBlob)
# ====================================================================
print("\n" + "=" * 65)
print(" STEP 4: Sentiment Analysis (TextBlob)")
print("=" * 65)

df['polarity'] = df['text'].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
df['subjectivity'] = df['text'].astype(str).apply(lambda t: TextBlob(t).sentiment.subjectivity)

def label_sentiment(p):
    if p > 0.1:
        return 'Positive'
    elif p < -0.1:
        return 'Negative'
    return 'Neutral'

df['sentiment'] = df['polarity'].apply(label_sentiment)

print(f"  Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}\n")

# ── 4a. Sentiment Distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
sent_counts = df['sentiment'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
axes[0].pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=140, textprops={'fontsize': 12})
axes[0].set_title("Overall Sentiment Distribution", fontweight='bold')

# Polarity histogram
axes[1].hist(df['polarity'], bins=40, color='mediumslateblue', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', ls='--', lw=1.5, label='Neutral (0)')
axes[1].set_title("Polarity Score Distribution", fontweight='bold')
axes[1].set_xlabel("Polarity")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_sentiment_distribution.png")
plt.close()
print("  ✓ Saved 06_sentiment_distribution.png")

# ── 4b. Polarity vs Subjectivity Scatter ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['polarity'], df['subjectivity'], alpha=0.3,
                     c=df['target'], cmap='tab10', s=15)
ax.set_xlabel("Polarity (Negative ← → Positive)")
ax.set_ylabel("Subjectivity (Objective ← → Subjective)")
ax.set_title("Polarity vs Subjectivity by Category", fontweight='bold')
ax.axhline(0.5, color='grey', ls=':', lw=1)
ax.axvline(0, color='grey', ls=':', lw=1)
handles, _ = scatter.legend_elements()
short_names = [c.split('.')[-1] for c in sorted(df['category'].unique())]
ax.legend(handles, short_names, loc='upper left', fontsize=9, title='Category')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_polarity_vs_subjectivity.png")
plt.close()
print("  ✓ Saved 07_polarity_vs_subjectivity.png")

# ── 4c. Mean Polarity by Category ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
mean_pol = df.groupby('category')['polarity'].mean().sort_values()
colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in mean_pol]
mean_pol.plot(kind='barh', color=colors_bar, ax=ax, edgecolor='white')
ax.set_title("Mean Sentiment Polarity by Category", fontweight='bold')
ax.set_xlabel("Mean Polarity")
ax.axvline(0, color='black', lw=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_mean_polarity_by_category.png")
plt.close()
print("  ✓ Saved 08_mean_polarity_by_category.png")

# ── 4d. Sentiment Breakdown per Category (Grouped Bar) ──────────────
fig, ax = plt.subplots(figsize=(12, 6))
ct = pd.crosstab(df['category'], df['sentiment'], normalize='index') * 100
ct.plot(kind='bar', stacked=True, color=['#e74c3c', '#95a5a6', '#2ecc71'],
        ax=ax, edgecolor='white')
ax.set_title("Sentiment Breakdown by Category (%)", fontweight='bold')
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_sentiment_by_category_stacked.png")
plt.close()
print("  ✓ Saved 09_sentiment_by_category_stacked.png")

# ====================================================================
# 5. TF-IDF VECTORIZATION
# ====================================================================
print("\n" + "=" * 65)
print(" STEP 5: TF-IDF Vectorization")
print("=" * 65)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), max_df=0.95, min_df=2)
X_tfidf = tfidf.fit_transform(df['clean_text'])

print(f"  TF-IDF matrix shape: {X_tfidf.shape}")
print(f"  Vocabulary size    : {len(tfidf.vocabulary_)}")

# ── 5a. Top TF-IDF terms per category ───────────────────────────────
fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5), sharey=False)
feature_names = np.array(tfidf.get_feature_names_out())

for i, cat in enumerate(sorted(df['category'].unique())):
    mask = df['category'] == cat
    mean_tfidf = X_tfidf[mask].mean(axis=0).A1
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    top_terms = feature_names[top_idx]
    top_scores = mean_tfidf[top_idx]

    axes[i].barh(range(len(top_terms)), top_scores[::-1],
                 color=sns.color_palette("coolwarm", 10))
    axes[i].set_yticks(range(len(top_terms)))
    axes[i].set_yticklabels(top_terms[::-1])
    axes[i].set_title(cat.split('.')[-1].title(), fontweight='bold')
    axes[i].set_xlabel("Mean TF-IDF")

fig.suptitle("Top 10 TF-IDF Terms per Category", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_tfidf_top_terms.png", bbox_inches='tight')
plt.close()
print("  ✓ Saved 10_tfidf_top_terms.png")

# ====================================================================
# 6. TOPIC MODELING (LDA)
# ====================================================================
print("\n" + "=" * 65)
print(" STEP 6: Topic Modeling (Latent Dirichlet Allocation)")
print("=" * 65)

N_TOPICS = 5
count_vec = CountVectorizer(max_features=3000, max_df=0.95, min_df=2)
X_counts = count_vec.fit_transform(df['clean_text'])

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42, max_iter=15,
    learning_method='online', n_jobs=-1
)
lda.fit(X_counts)

cv_feature_names = count_vec.get_feature_names_out()

print(f"  Discovered {N_TOPICS} topics:\n")
topic_labels = []
for idx, topic in enumerate(lda.components_):
    top_words = [cv_feature_names[i] for i in topic.argsort()[-8:][::-1]]
    topic_labels.append(f"Topic {idx + 1}")
    print(f"  Topic {idx + 1}: {', '.join(top_words)}")

# ── 6a. Topic–Word heatmap ──────────────────────────────────────────
n_top_words = 10
topic_word_matrix = np.zeros((N_TOPICS, n_top_words))
word_labels = []

for idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    topic_word_matrix[idx] = topic[top_indices]
    if idx == 0:
        word_labels = [cv_feature_names[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(12, 6))
# Build a label matrix for all topics
all_top_words = []
for idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    all_top_words.append([cv_feature_names[i] for i in top_indices])

# Use unique words across all topics
unique_words = []
for wlist in all_top_words:
    for w in wlist:
        if w not in unique_words:
            unique_words.append(w)
unique_words = unique_words[:20]  # limit

heat_data = np.zeros((N_TOPICS, len(unique_words)))
for i, topic in enumerate(lda.components_):
    for j, w in enumerate(unique_words):
        widx = list(cv_feature_names).index(w) if w in cv_feature_names else -1
        if widx >= 0:
            heat_data[i, j] = topic[widx]

sns.heatmap(heat_data, xticklabels=unique_words,
            yticklabels=[f"Topic {i+1}" for i in range(N_TOPICS)],
            cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title("LDA Topic–Word Importance Heatmap", fontweight='bold')
ax.set_xlabel("Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_lda_topic_heatmap.png")
plt.close()
print("\n  ✓ Saved 11_lda_topic_heatmap.png")

# ── 6b. Document-Topic distribution ────────────────────────────────
doc_topic_dist = lda.transform(X_counts)
df['dominant_topic'] = doc_topic_dist.argmax(axis=1) + 1

fig, ax = plt.subplots(figsize=(8, 5))
topic_counts = df['dominant_topic'].value_counts().sort_index()
ax.bar(topic_counts.index, topic_counts.values,
       color=sns.color_palette("Set2", N_TOPICS), edgecolor='white')
ax.set_title("Document Count by Dominant Topic", fontweight='bold')
ax.set_xlabel("Topic")
ax.set_ylabel("Number of Documents")
ax.set_xticks(range(1, N_TOPICS + 1))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_document_topic_distribution.png")
plt.close()
print("  ✓ Saved 12_document_topic_distribution.png")

# ====================================================================
# 7. TEXT CLASSIFICATION
# ====================================================================
print("\n" + "=" * 65)
print(" STEP 7: Text Classification (NB, LR, SVM)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, df['target'], test_size=0.2, random_state=42, stratify=df['target']
)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'y_pred': y_pred,
    }
    print(f"\n  {name}:")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    Recall    = {rec:.4f}")
    print(f"    F1-Score  = {f1:.4f}")

# ── 7a. Model Comparison Bar Chart ──────────────────────────────────
metrics_df = pd.DataFrame({
    k: {m: v for m, v in vals.items() if m != 'y_pred'}
    for k, vals in results.items()
}).T

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='white', width=0.8)
ax.set_title("Model Performance Comparison", fontweight='bold')
ax.set_ylabel("Score")
ax.set_ylim(0.5, 1.02)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc='lower right')
# add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_model_comparison.png")
plt.close()
print("\n  ✓ Saved 13_model_comparison.png")

# ── 7b. Confusion Matrices ─────────────────────────────────────────
short_labels = [c.split('.')[-1] for c in newsgroups.target_names]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, vals) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, vals['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=short_labels, yticklabels=short_labels)
    ax.set_title(f"{name}\n(Acc: {vals['Accuracy']:.3f})", fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_confusion_matrices.png", bbox_inches='tight')
plt.close()
print("  ✓ Saved 14_confusion_matrices.png")

# ── 7c. Classification Report for best model ───────────────────────
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
best_preds = results[best_model_name]['y_pred']

print(f"\n  Best Model: {best_model_name}")
print(f"\n  Classification Report ({best_model_name}):")
report_str = classification_report(
    y_test, best_preds, target_names=newsgroups.target_names
)
print(report_str)

# Parse classification report for a heatmap
report_dict = classification_report(
    y_test, best_preds, target_names=short_labels, output_dict=True
)
report_df = pd.DataFrame(report_dict).iloc[:3].T.drop(
    ['accuracy', 'macro avg', 'weighted avg'], errors='ignore'
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(report_df.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, vmin=0.5, vmax=1)
ax.set_title(f"Classification Report Heatmap – {best_model_name}", fontweight='bold')
ax.set_ylabel("Category")
ax.set_xlabel("Metric")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_classification_report_heatmap.png")
plt.close()
print("  ✓ Saved 15_classification_report_heatmap.png")

# ====================================================================
# 8. SUMMARY DASHBOARD
# ====================================================================
print("\n" + "=" * 65)
print(" STEP 8: Generating Summary Dashboard")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

# (0,0) Category distribution
ax1 = fig.add_subplot(gs[0, 0])
df['category'].value_counts().plot(kind='bar', ax=ax1, color=sns.color_palette("viridis", n_cats))
ax1.set_title("Category Distribution", fontweight='bold', fontsize=11)
ax1.set_xticklabels([c.split('.')[-1] for c in sorted(df['category'].unique())],
                     rotation=30, ha='right', fontsize=8)
ax1.set_ylabel("Count")

# (0,1) Sentiment pie
ax2 = fig.add_subplot(gs[0, 1])
sent_counts = df['sentiment'].value_counts()
ax2.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c', '#95a5a6'], startangle=140)
ax2.set_title("Sentiment Distribution", fontweight='bold', fontsize=11)

# (0,2) Polarity Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['polarity'], bins=30, color='slateblue', edgecolor='white')
ax3.axvline(0, color='red', ls='--', lw=1)
ax3.set_title("Polarity Scores", fontweight='bold', fontsize=11)
ax3.set_xlabel("Polarity")

# (1,0) Word Count Box
ax4 = fig.add_subplot(gs[1, 0])
sns.boxplot(data=df, x='category', y='word_count', hue='category',
            palette='Set2', legend=False, ax=ax4)
ax4.set_xticklabels([c.split('.')[-1] for c in sorted(df['category'].unique())],
                     rotation=30, ha='right', fontsize=8)
ax4.set_title("Word Count by Category", fontweight='bold', fontsize=11)

# (1,1) Top 10 words
ax5 = fig.add_subplot(gs[1, 1])
top10 = Counter(all_words).most_common(10)
w10, c10 = zip(*top10)
ax5.barh(list(reversed(w10)), list(reversed(c10)), color=sns.color_palette("magma_r", 10))
ax5.set_title("Top 10 Words", fontweight='bold', fontsize=11)

# (1,2) Model comparison
ax6 = fig.add_subplot(gs[1, 2])
model_accs = {k: v['Accuracy'] for k, v in results.items()}
bars_dash = ax6.bar(model_accs.keys(), model_accs.values(),
                     color=['#3498db', '#e67e22', '#e74c3c'], edgecolor='white')
ax6.set_title("Model Accuracy Comparison", fontweight='bold', fontsize=11)
ax6.set_ylim(0.7, 1.0)
for b in bars_dash:
    ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f'{b.get_height():.3f}', ha='center', fontweight='bold', fontsize=9)

# (2, 0:3) Best model confusion matrix
ax7 = fig.add_subplot(gs[2, :])
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
            xticklabels=short_labels, yticklabels=short_labels)
ax7.set_title(f"Best Model ({best_model_name}) – Confusion Matrix", fontweight='bold', fontsize=12)
ax7.set_xlabel("Predicted")
ax7.set_ylabel("Actual")

fig.suptitle("📊 TEXT ANALYSIS ML PROJECT – SUMMARY DASHBOARD",
             fontsize=18, fontweight='bold', y=1.01)
plt.savefig(f"{OUTPUT_DIR}/16_summary_dashboard.png", bbox_inches='tight')
plt.close()
print("  ✓ Saved 16_summary_dashboard.png")

# ====================================================================
# DONE
# ====================================================================
print("\n" + "=" * 65)
print(f" ✅ ALL DONE! {16} visualizations saved to ./{OUTPUT_DIR}/")
print("=" * 65)
print(f"\n  Generated plots:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        print(f"    • {f}")
print()
