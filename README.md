End-to-end NLP pipeline that classifies tweets as **Positive**, **Negative**, or **Neutral**
using three models of increasing complexity on **74,000+ real tweets**.

## Models Used

| Model | Approach | Library |
|---|---|---|
| TF-IDF + Logistic Regression | Traditional ML | Scikit-learn |
| Bidirectional LSTM | Deep Learning | PyTorch |
| BERT | Transformer | HuggingFace |

# Results

| Model | Accuracy |
|---|---|
| TF-IDF + LogReg | ~82% |
| BiLSTM | ~85% |
| BERT | ~89% |
