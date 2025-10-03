# Task 1 — Correlation Analysis

## 📌 Assignment
We were instructed to:
- Find the data at `max.ge/aiml_midterm/829461_html`.  
- Compute **Pearson’s correlation coefficient** for the given points.  
- Provide a **graph** to visualize the relationship.

Since the exam system only displayed the chart interactively, the **coordinates were manually extracted** from the graph by hovering over each blue dot.

---

## 📊 Data Points
The dataset (x, y) is:

(-5, -5), (-5, 2), (-3, -1), (-1, 1),
(1, -2), (3, 1), (5, -3), (7, -2)

## ⚙️ Method
1. **Extract points** manually from the online graph.  
2. Implement Pearson correlation coefficient:  

   \[
   r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}
   \]

3. Fit a best-fit regression line:

   \[
   y = mx + b
   \]

4. Plot the scatter diagram with matplotlib, add regression line.  


Results

Pearson’s correlation coefficient:

𝑟≈−0.299

→ weak negative correlation.

Regression line:

𝑦≈−0.149𝑥−0.987


The analysis shows that X and Y have a weak negative correlation. As X increases, Y tends to decrease slightly, but the effect is not strong. The scatter plot illustrates this relationship.