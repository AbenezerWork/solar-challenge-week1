# Solar Challenge - Week 1

Welcome to **Week 1** of the Solar Challenge!  
This repository contains my analysis of solar energy production across **Benin**, **Sierra Leone**, and **Togo**, with a focus on:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Dashboard Development

---

## 📦 Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AbigailF1/solar-challenge-week1.git
cd solar-challenge-week1

```

### 2. Create & Activate a Virtual Environment

You can set up your Python environment using either `venv` or `conda`:

#### 🔹 Using `venv` (built-in Python module)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### 🔹 Using conda (if you have Anaconda/Miniconda)

```bash
conda create -n solar-challenge python=3.10
conda activate solar-challenge
```

### 3. Install Required Packages

Once your virtual environment is activated, install all the required Python packages using:

```bash
pip install -r requirements.txt
```

This command will install all the dependencies listed in the requirements.txt file so you can run the notebooks and scripts without issues.

---

## 🗂️ Folder Structure

Below is an overview of the project's folder structure:

```bash
solar-challenge-week1/
.
├── README.md
├── app
│   ├── __pycache__
│   │   └── utils.cpython-313.pyc
│   ├── main.py
│   └── utils.py
├── data
│   ├── benin-malanville_clean.csv
│   ├── sierraleone-bumbuna_clean.csv
│   ├── solar-measurements_benin-malanville_qc.csv
│   ├── solar-measurements_sierraleone-bumbuna_qc.csv
│   ├── solar-measurements_togo-dapaong_qc.csv
│   └── togo-dapaong_clean.csv
├── notebooks
│   ├── benin-eda.ipynb
│   ├── compare_countries.ipynb
│   ├── sierraleone.ipynb
│   └── togo-eda.ipynb
└── requirements.txt

5 directories, 15 files

```

---

## 🚀 How to Use

You can explore each country's solar energy data through the Jupyter notebooks located in the `notebooks/` folder:

- `benin_eda.ipynb`
- `SierraLeone_eda.ipynb`
- `togo_eda.ipynb`

Each notebook includes:

- ✅ Basic data inspection  
- 🧹 Handling missing values  
- 📈 Time series visualization  
- 📊 Statistical summaries  
- 📅 Daily and monthly trends  

Simply open a notebook with Jupyter


---

## 🛠 Tech Stack

This project uses the following tools and libraries:

- **Python 3.10** – Programming language
- **Jupyter Notebooks** – Interactive coding environment
- **Pandas**, **NumPy** – Data manipulation and numerical operations
- **Matplotlib**, **Seaborn** – Visualization and plotting libraries

These tools provide a strong foundation for conducting data analysis and creating insightful visuals.



