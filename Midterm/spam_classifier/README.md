# Task 2 — Spam Classifier

## Problem Statement
We are given a dataset of email features in CSV format (`m_tsirekidze2024_829461.csv`) and asked to build a spam classifier.  
The dataset contains numeric features extracted from emails, such as:

- **words** → total number of words in the email  
- **links** → number of hyperlinks  
- **capital_words** → number of capitalized words  
- **spam_word_count** → number of spam-related keywords  

The task is to train a model, evaluate its performance, and provide reproducible results.

---

## Steps Taken

### 1. Training the Model
We implemented a training script: **`train_and_eval.py`**.  
It loads the dataset, splits it into train/test sets, trains a logistic regression classifier, and saves the model + schema.

Run the following command to train and evaluate:

```bash
python spam_classifier/train_and_eval.py \
  --csv spam_classifier/m_tsirekidze2024_829461.csv \
  --test_size 0.30 \
  --random_state 42


2. Results

On the test set, the model achieved:

✅ Accuracy: 0.9613 (96.13%)

✅ Saved trained model in: artifacts/model.joblib

✅ Saved schema (features): artifacts/schema.json

3. Using the Classifier (CLI)

We also implemented a simple command-line interface (app_cli.py) that loads the trained model and classifies new emails based on features.

Example usage:

python spam_classifier/app_cli.py \
  --words 120 \
  --links 3 \
  --capital_words 5 \
  --spam_word_count 2


Output:

[INPUT] words=120, links=3, capital_words=5, spam_word_count=2
[PREDICTION] This email is: SPAM

4. Visualizations

We included two important plots:

Confusion Matrix

Shows how well the classifier distinguishes between Spam and Ham (Legit) emails.

Feature Importance

Shows which features influence the spam decision the most. For example, spam_word_count and links are usually strong predictors.

Run the script to generate plots:

python spam_classifier/train_and_eval.py \
  --csv spam_classifier/m_tsirekidze2024_829461.csv \
  --test_size 0.30 \
  --random_state 42 \
  --plot


Plots will be saved under artifacts/.

Conclusion

1. The spam classifier reached 96.13% accuracy, which is a strong result.

2. The project is fully reproducible with the provided code and dataset.

3. The trained model and schema are stored in artifacts/ for later use.

4. CLI allows fast classification of new email examples.