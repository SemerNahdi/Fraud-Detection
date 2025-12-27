# Multi-Domain Fraud Detection using DeBERTa and LoRA 

This project implements a robust fraud detection system designed to identify deceptive content across multiple domains (phishing, fake news, product reviews, etc.). By leveraging the power of **DeBERTa-v3** and **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**, we achieve state-of-the-art performance while maintaining computational efficiency.

---

## 💾 Dataset Link

Access the comprehensive dataset used for this project on Hugging Face:

🔗 **[DIFraud Dataset - Hugging Face](https://huggingface.co/datasets/redasers/difraud?utm_source=chatgpt.com)**

---

## 🚀 Ecosystem & Resources

Explore the complete Fraud Detection ecosystem:

*   **🖥️ Frontend Dashboard**: [PhantomRisk Frontend](https://github.com/SemerNahdi/Fraud-Dector-Frontend---PhantomRisk)
*   **🔌 Browser Extension**: [Gmail Phishing Extension](https://github.com/SemerNahdi/gmail-phishing-extension)
*   **📄 Product Documentation**: [Fraud Detector Overview (PDF)](./Fraud%20detector%20.pdf)

---



## 🔍 1. Understanding Steps

The following steps were taken to explore and understand the dataset's characteristics before model development:

1.  **Data Ingestion**: Loading diverse JSONL datasets covering 7 distinct domains:
    *   Phishing emails
    *   Fake news articles
    *   Political statements
    *   Product reviews
    *   Job scams
    *   SMS/Text scams
    *   Twitter rumors
2.  **General Statistics**: Analyzed sample distributions, deceptive vs. non-deceptive ratios, and average text lengths per domain.
3.  **Visual Exploratory Data Analysis (EDA)**:
    *   **Label Distribution**: Identified class imbalances (notably in Job Scams).
    *   **Text Length Analysis**: Studied character and word counts across labels.
    *   **Word Frequency & N-grams**: Extracted top meaningful words after removing custom stopwords (e.g., "http", "click", "email").
    *   **Word Clouds**: Visualized semantic differences between deceptive and legitimate content.
    *   **Sentiment Analysis**: Used VADER to compare emotional tones across different domains.
    *   **Vocabulary Richness**: Measured the ratio of unique words to total words to detect linguistic patterns in fraud.

---

## ⚙️ 2. Processing Steps

Data preparation focused on cleaning and feature engineering to enhance model sensitivity:

1.  **Text Cleaning & Normalization**:
    *   Stripped HTML tags using `BeautifulSoup`.
    *   Normalized identifiers: Replaced specific URLs, emails, and emojis with tokens like `<URL>`, `<EMAIL>`, and `<EMOJI>`.
    *   Standardized punctuation and whitespace for better tokenization.
2.  **Semantic Keyword Expansion**:
    *   Generated an expanded fraud cue list using `SentenceTransformers`. Base keywords like "urgent" or "winner" were semantically expanded to include related terms, increasing detection coverage.
3.  **Feature Engineering**:
    *   Extracted counts for URLs, emails, and expanded suspicious keywords.
    *   Added binary indicators for the presence of fraud cues.
4.  **Tokenization**:
    *   Utilized **DeBERTa-v3** tokenizer with **stride-based overlapping** for long documents, ensuring no critical information is lost during truncation.
5.  **Class Weight Calibration**:
    *   Computed balanced class weights to mitigate the impact of highly imbalanced datasets during training.

---

## 🏗️ 3. Model Architecture

The core architecture utilizes a **Multi-Task Learning** approach combined with **LoRA adapters** for maximum adaptability.

### Backbone: DeBERTa-v3
We use **Microsoft's DeBERTa-v3** as the backbone, which improves upon BERT/RoBERTa using disentangled attention and an enhanced mask decoder.

### Parameter-Efficient Fine-Tuning (LoRA)
Instead of fine-tuning millions of parameters, we inject **Low-Rank Adapters (LoRA)** into the attention matrices (`query_proj`, `value_proj`, `key_proj`). This significantly reduces the memory footprint and prevents catastrophic forgetting.

### Multi-Task Design
To encourage the model to learn domain-agnostic fraud features, we implemented a dual-head architecture:
1.  **Shared Adapter**: Captures cross-domain deceptive patterns.
2.  **Fraud Classifier Head**: Binary classification (Real vs. Deceptive).
3.  **Domain Classifier Head**: Classifies the source domain (e.g., "SMS" vs. "Fake News"), forcing the backbone to learn high-level features that generalize across contexts.

**Loss Function**:
$$Loss = 0.7 \times \text{Fraud\_Loss} + 0.3 \times \text{Domain\_Loss}$$
![Fraud Detection Architecture Overview](./Architecture.png)
---

## 🖼️ Project Visuals

![Fraud Detection Architecture Overview](./image.png)


---
