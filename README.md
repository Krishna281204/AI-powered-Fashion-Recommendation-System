# 🛍️ AI-Powered Fashion Recommender System

## 📌 Overview

This project is a **Hybrid Recommendation System** that suggests fashion products by combining **content-based filtering** and **popularity-based ranking**.

The system analyzes product attributes such as brand, color, description, and price category to recommend relevant and similar items. It also incorporates product popularity to improve the overall quality of recommendations.

---

## 🚀 Features

* 🔹 Content-based filtering using **TF-IDF**
* 🔹 Popularity-based ranking with **log scaling**
* 🔹 Hybrid recommendation with adjustable weight (α)
* 🔹 Price segmentation (Low / Medium / High)
* 🔹 Optional **same-brand recommendation filter**
* 🔹 **Explainable recommendations** using:

  * Match Score (content similarity)
  * Popularity Score
* 🔹 Interactive UI built with Streamlit

---

## 🧠 How It Works

### 1. Data Preprocessing

* Handled missing values
* Cleaned and standardized product data
* Created a **price bucket feature** (low / medium / high)

### 2. Feature Engineering

* Combined multiple attributes:

  * brand
  * colour
  * description
  * product attributes
  * price category
* Converted text into numerical form using **TF-IDF Vectorization**

### 3. Similarity Computation

* Computed similarity between products using **linear kernel**
* Efficient alternative to cosine similarity

### 4. Popularity Score

Calculated using:

```
popularity = avg_rating × log(1 + rating_count)
```

* Helps avoid bias toward extremely popular products
* Normalized before combining

### 5. Hybrid Recommendation

Final score:

```
Hybrid Score = α × Content Similarity + (1 − α) × Popularity Score
```

* α is adjustable via UI slider

---

## 🖥️ Output Example

Each recommendation shows:

* Product name and brand
* Price and rating
* **Match Score (similarity)**
* **Popularity Score**

This makes the system **transparent and explainable**

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Framework:** Streamlit

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
fashion-recommender-system/
│── app.py
│── requirements.txt
│── README.md
│── Fashion Dataset.csv (or external link)
```

---

## 🎯 Key Highlights

* Implemented a **hybrid recommendation system**
* Improved ranking using **log-scaled popularity**
* Applied **feature engineering (price segmentation)**
* Built an **interactive and explainable UI**
* Handled real-world issues like **index alignment and data consistency**

---

## 📌 Future Improvements

* User-based recommendations
* Evaluation metrics (Precision@K, Recall@K)
* Deployment on cloud (Streamlit Cloud / Render)
* Add product images for enhanced UI

---

## 👤 Author

Krishna Sharma