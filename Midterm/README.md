# AI & ML for Cybersecurity — Midterm (Mariam Tsirekidze)

**Repository purpose:** Contains my reproducible answers and code for both exam tasks.  
**Language:** Python.  
**How to reproduce:** See the quickstart in each task section.

---

## Task 1 — Correlation from Online Graph (10 pts)

**Source:** The exam page shows blue points; their coordinates appear on hover.  
Because the page only displays a chart (not a raw dataset), I manually read each point by hovering and recorded them in code.

### Data (manually read from the online graph)
```text
(-5, -5), (-5, 2), (-3, -1), (-1, 1),
(1, -2), (3, 1), (5, -3), (7, -2)



## 📝 Task 2 — Spam Classifier
- Dataset: `m_tsirekidze2024_829461.csv` (features extracted from emails).  
- Implemented a spam classifier with **Logistic Regression** using scikit-learn.  
- Training pipeline includes:
  - CSV loading with Pandas  
  - Train/test split  
  - Accuracy evaluation  
  - Saving artifacts (`model.joblib`, `schema.json`)  
- Added a **CLI tool** for classifying new emails.  

**Result:**  
- Accuracy on test set: **96.13%**  
- Model + schema saved for reuse.  

📍 See full details in [spam_classifier/README.md](spam_classifier/README.md).


## ⚙️ How to Reproduce

1. clone Repo. 

2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run Task 1 (Correlation)
python correlation/correlation.py


Output:

Pearson correlation coefficient printed

Scatter plot saved as Plot.png

4. Run Task 2 (Spam Classifier)

Train + evaluate:

python spam_classifier/train_and_eval.py \
  --csv spam_classifier/m_tsirekidze2024_829461.csv \
  --test_size 0.30 \
  --random_state 42


Use CLI for prediction:

python spam_classifier/app_cli.py \
  --words 100 --links 2 --capital_words 5 --spam_word_count 3

✅ Conclusion

Task 1: Implemented Pearson correlation manually and visualized results.

Task 2: Built a spam classifier achieving 96% accuracy with reproducible training + evaluation.

The repository is structured for clarity and reproducibility, with detailed task-level READMEs.
