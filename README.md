# Adult Income Classifier — MLOps Pipeline

An end-to-end MLOps project that trains a binary income classifier on the UCI Adult dataset and deploys it as a live REST API on AWS. Built to demonstrate the full ML lifecycle: data processing, model training, experiment tracking, containerization, CI/CD, and cloud deployment.

**Live API:** `http://23.22.30.181:8000/docs`

---

## Architecture

```
UCI Adult Dataset (Hugging Face)
        │
        ▼
  Data Loading & EDA
  (sweetviz report, class imbalance analysis)
        │
        ▼
  Feature Engineering Pipeline (sklearn)
  (log transform, ordinal/one-hot encoding, feature engineering)
        │
        ▼
  Model Training + MLflow Tracking
  (Logistic Regression vs Gradient Boosting)
        │
        ▼
  FastAPI REST API  ──►  Docker Image  ──►  Amazon ECR
                                                │
                                                ▼
                                          EC2 t2.micro
                                       (live predictions)
```

---

## Results

| Model | F1 Score | AUC-ROC |
|---|---|---|
| Logistic Regression | baseline | baseline |
| **Gradient Boosting** | **0.6973** | **0.9248** |

Dataset: 32,561 rows, 15 features, 76/24 class imbalance (≤50K / >50K income)

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | Hugging Face Datasets (`jlh/uci-adult-income`) |
| EDA | sweetviz |
| Feature Engineering | scikit-learn Pipeline |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | Amazon ECR |
| Compute | Amazon EC2 (t2.micro) |

---

## Project Structure

```
adult-income-mlops/
├── src/
│   ├── data_loader.py       # Loads dataset from Hugging Face
│   ├── features.py          # sklearn preprocessing pipeline
│   └── train.py             # Model training + MLflow logging
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # Request/response schemas
│   └── model.pkl            # Saved best model
├── notebooks/
│   └── 01_eda.ipynb         # EDA + sweetviz report
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Feature Engineering

The sklearn pipeline applies the following transformations:

- **log1p transform** on `capital-gain` and `capital-loss` (skew > 4)
- **OrdinalEncoder** on `education`
- **OneHotEncoder** on remaining categorical features
- **Drops** `education-num` (redundant with `education`)
- **Engineers** `capital_diff` and `age_group` features

---

## API Usage

**Endpoint:** `POST /predict`

Example request:
```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

Example response:
```json
{
  "prediction": 0,
  "probability": 0.0818
}
```

`prediction: 0` = income ≤50K, `prediction: 1` = income >50K

Interactive docs available at `/docs`.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/davyjones7321/adult-income-mlops
cd adult-income-mlops

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# View MLflow experiment results
mlflow ui  # opens at localhost:5000

# Start the API
uvicorn api.main:app --reload  # opens at localhost:8000
```

---

## Running with Docker

```bash
docker build -t adult-income-api .
docker run -p 8000:8000 adult-income-api
```

---

## CI/CD

GitHub Actions runs on every push to `main`:
- Validates Python imports
- Tests API schema
- Status: ![CI](https://github.com/davyjones7321/adult-income-mlops/actions/workflows/ci.yml/badge.svg)

---

## AWS Deployment

The Docker image is pushed to **Amazon ECR** and pulled onto an **EC2 t2.micro** instance running Amazon Linux 2023.

```
Local Docker Image
      │
      ▼
Amazon ECR (private registry)
      │
      ▼
EC2 t2.micro ──► docker run -p 8000:8000
      │
      ▼
Live API at http://23.22.30.181:8000
```

---

## MLflow Tracking

Best run: `0666cf0cf6fa40a28fe6c35a14916e68`

Tracked per experiment:
- F1 Score
- AUC-ROC
- Model artifact (saved as `model.pkl`)
