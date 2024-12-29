# SMS Spam Detection System 🚫✉️

## 📝 Project Overview
The **SMS Spam Detection** project aims to automatically classify text messages as **spam** or **ham** (not spam) using various machine learning techniques. It involves preprocessing SMS data, transforming the text into numerical features, and training multiple models to predict whether a message is spam. This project can be extended for use in applications like spam filtering for emails, SMS, and messaging services.

## 🔧 Key Components
- **Data Preprocessing**: Cleaning, tokenizing, and normalizing SMS text data.
- **Feature Extraction**: Using **TF-IDF** vectorization to convert text into numerical representations.
- **Model Training**: Comparing multiple machine learning models to find the best-performing one.
- **Performance Evaluation**: Using metrics like **accuracy**, **precision**, **recall**, and **F1-score** for evaluation.

### 🎥 Demo
Here's an example of how the model works:

| SMS Message                                             | Prediction |
| ------------------------------------------------------- | ---------- |
| "Congratulations! You've won a $1000 gift card!"        | Spam       |
| "Hey, can we reschedule our meeting to tomorrow?"       | Ham        |

## ✨ Features
- **Real-time SMS Spam Detection**: Predict whether an incoming message is spam or ham.
- **Multiple Model Comparison**: Evaluate various machine learning algorithms for classification.
- **Visualization Tools**: Graphs and plots to better understand the model performance.
- **Pretrained Models**: Use saved models for real-time predictions on new data.
- **Text Preprocessing Techniques**: Convert raw text into meaningful features for model training.

## 📊 Dataset
The dataset used is the **SMS Spam Collection Dataset**, which consists of approximately **5,572 SMS messages** labeled as "spam" or "ham." Each message is preprocessed to remove punctuation, convert to lowercase, and handle stopwords.

### Dataset Columns:
| Column   | Description                              |
| -------- | ---------------------------------------- |
| **Label** | Indicates if the message is spam or ham |
| **Message** | The actual text of the SMS message    |

## 🧰 Technologies Used
### Languages and Libraries:
- **Python 3.6+**: The programming language used.
- **scikit-learn**: For machine learning algorithms and model evaluation.
- **pandas**: Data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib & seaborn**: Data visualization and plotting.
- **nltk**: Text preprocessing and tokenization.


