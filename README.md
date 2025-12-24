# 🎬 Movie Review Sentiment Analysis using RNN

## 📌 Overview
This project performs **sentiment analysis on movie reviews** using a **Recurrent Neural Network (RNN)**.  
It predicts whether a given movie review is **Positive** or **Negative** based on its textual content.

The model is trained on the **IMDB Movie Review Dataset** and deployed using **Streamlit** for real-time, interactive predictions.
---
## 🔗 Live Demo
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://moviereview-sentimentanalysis-c7zaqpkvasjmaavhrhqeeb.streamlit.app/)

---
## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
---
## ✨ Key Features
- Text preprocessing and tokenization
- Sequence padding for uniform input length
- Simple Recurrent Neural Network (RNN) model
- Binary sentiment classification (Positive / Negative)
- Interactive Streamlit web application
- Real-time prediction with confidence score
---
## 📊 Dataset
The project uses the **IMDB Movie Review Dataset**, which contains labeled movie reviews for binary sentiment classification.  
Reviews are converted into numerical sequences using a predefined word index, allowing the model to learn sentiment patterns from text.

---
## 🚀 How to Run the Project
```bash
# Clone the repository
git clone https://github.com/iskushpatel/Movie_review-sentiment_analysis.git

# Navigate to the project directory
cd Movie_review-sentiment_analysis

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
````
---
## 🚀 Model Architecture
- **Embedding Layer** – converts words into dense vector representations  
- **Simple RNN Layer** – captures sequential patterns in text  
- **Dense Output Layer** – sigmoid activation for sentiment prediction  

**Optimizer:** Adam  
**Loss Function:** Binary Cross-Entropy  

---
## 📊 Model Performance
- **Training Accuracy:** ~86%  
- **Validation Accuracy:** ~85%  
- **Training Loss:** Decreases consistently  
### Prediction Logic
- **Probability > 0.5** → Positive  
- **Probability ≤ 0.5** → Negative  
The model performs consistently on unseen reviews and user-provided input.
