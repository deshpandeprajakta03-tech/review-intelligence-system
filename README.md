# 🧠 Review Intelligence System

An AI-powered product review classifier that detects **sentiment** and **issue type** from user reviews using fine-tuned **RoBERTa** transformer models.

---

## 🚀 Demo

![App Screenshot](screenshots/demo.png)

---

## 📌 Features

- Classifies reviews as **Positive** or **Negative**
- Identifies **8 issue categories** for negative reviews:
  - App Stability Issue
  - Audio / Voice Issue
  - Connection or Access Issue
  - Performance / Slowness
  - Recording / Playback Issue
  - Screen Sharing Issue
  - User Interface / Usability Issue
  - Video Quality Issue
- Built with **RoBERTa** (roberta-base) fine-tuned on real Zoom app reviews
- Deployed as a **Flask** web application

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | RoBERTa (HuggingFace Transformers) |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Training | PyTorch, HuggingFace Trainer API |
| Data | Zoom App Reviews Dataset |

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Sentiment Classification | 88% |
| Issue Classification | 82% |

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/deshpandeprajakta03-tech/review-intelligence-system.git
cd review-intelligence-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install accelerate>=1.1.0
```

### 4. Train the models
```bash
python -m training.train_sentiment_roberta
python -m training.train_issue_roberta
```

### 5. Run the app
```bash
python main.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

```
review-intelligence-system/
├── app/
│   ├── classifier.py       # Model loading and prediction
│   ├── trainer.py          # Training utilities
│   └── utils.py            # Text preprocessing
├── data/
│   └── zoom_classification_dataset_final.xlsx
├── models/                 # Saved models (generated after training)
├── templates/
│   └── index.html          # Web UI
├── training/
│   ├── preprocessing.py
│   ├── train_sentiment_roberta.py
│   └── train_issue_roberta.py
├── main.py                 # Flask app entry point
└── requirements.txt
```

---

## 🧑‍💻 Author

**Prajakta Deshpande**  
[GitHub](https://github.com/deshpandeprajakta03-tech)
