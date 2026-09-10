from app.trainer import train_sentiment_model, train_issue_model

# Define file path
excel_path = "data/zoom_classification_dataset_final.xlsx"

# Train both models
train_sentiment_model(excel_path, output_dir="models/sentiment_model")
train_issue_model(excel_path, output_dir="models/roberta_issue_model")
