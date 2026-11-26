# FinMind — AI-Powered Financial Intelligence Platform

FinMind is a deep-learning-based, multi-agent financial intelligence platform designed to help small and medium-sized businesses (SMEs) understand their finances, monitor transactions, forecast cash flow, and identify unusual financial activity.

## Project Overview

FinMind brings financial data management, analytics, machine-learning insights, and an AI assistant into one platform. It is designed to support clearer financial decision-making through accessible dashboards and intelligent analysis.

## Core Capabilities

- **Financial management:** manage accounts and transactions.
- **Cash-flow insights:** review financial activity and cash-flow trends.
- **Forecasting:** generate predictive insights from historical financial data.
- **Anomaly detection:** identify potentially unusual transactions or patterns.
- **Analytics dashboard:** view financial summaries and indicators.
- **AI financial assistant:** interact with financial information using natural-language questions.
- **Document intelligence:** support workflows involving financial documents.

> The availability of each capability depends on the current implementation and configured services.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Backend API | Python, FastAPI |
| Database | PostgreSQL, pgvector |
| Machine Learning | PyTorch; time-series and anomaly-detection components |
| AI integration | Gemini API and Ollama (where configured) |
| Migrations | Alembic |
| Deployment | Vercel for the frontend; a separate host is required for the API and database |

## Repository Structure

```text
FinMind/
├── alembic/          # Database migration configuration
├── backend/          # FastAPI application and services
├── data/             # Data assets, schemas, models, and manifests
├── docs/             # Project documentation
├── frontend/         # Next.js application
├── ml/               # Machine-learning code
├── scripts/          # Setup, ingestion, and training utilities
├── requirements.txt  # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vishnu1333/FinMind.git
cd FinMind
```

### 2. Configure environment variables

Create local environment files using the example files provided by the project, where available. Add database connection details and required API credentials locally.

**Do not commit `.env` files, passwords, API keys, or other secrets.**

### 3. Set up the backend

Use the project's Python requirements and setup instructions:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Review `START_COMMANDS.txt`, the backend documentation, and migration configuration for the exact database and server startup commands for your environment.

### 4. Set up the frontend

```bash
cd frontend
npm install
```

Use the scripts defined in `frontend/package.json` to run or build the application.

## Deployment Notes

- Deploy the **frontend** from the `frontend` directory on Vercel.
- Deploy the **FastAPI backend** separately on a Python-compatible hosting platform.
- Provision PostgreSQL (and pgvector if required) separately.
- Configure the frontend API URL, backend CORS origins, database credentials, and AI provider credentials in the appropriate hosting environment variables.
- Never put private backend credentials in client-exposed variables.

## Data and Model Assets

Large raw datasets, generated outputs, local virtual environments, dependency folders, and secrets should be stored outside Git where appropriate. Follow the repository's `.gitignore` rules and the data instructions in the project documentation.

## Project Status

This repository contains the FinMind project source. Features that rely on external services, database setup, datasets, or trained model artifacts may require additional configuration before they can run.

## Author

**Vishnu Vardhan Chilakacharla**

## License

No license has been specified yet. Unless a license is added, the repository's contents remain subject to the applicable default copyright rules.
