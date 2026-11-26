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


## Gmail OTP Authentication

FinMind registration and every login use email OTP verification. The API sends a six-digit code through Gmail SMTP; codes expire after five minutes, are single-use, and allow up to five verification attempts. Resend is throttled and limited per email and purpose.

Configure these variables in the **backend** environment (local `.env` or your backend host's secret settings):

- `SMTP_EMAIL`: dedicated Gmail sender address
- `SMTP_APP_PASSWORD`: a newly generated Google App Password (remove spaces)
- `SMTP_HOST`: `smtp.gmail.com` (default)
- `SMTP_PORT`: `587` (default; STARTTLS)
- `SMTP_FROM_NAME`: `FinMind Security` (default)
- `OTP_HASH_SECRET`: random secret of at least 32 characters

Generate a secret locally:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the generated value into your backend environment as `OTP_HASH_SECRET`. Do not commit real credentials. The checked-in `.env.example` contains placeholders only. If a Gmail App Password was previously shared or exposed, revoke it and generate a replacement.

The OTP challenge table is created by the backend's SQLAlchemy metadata initialization. Existing deployments should restart the backend after deploying this change so the table is created.

Authentication endpoints:

- `POST /auth/register` — submit registration details and email a registration OTP
- `POST /auth/register/verify` — verify registration OTP and create the account
- `POST /auth/login` — validate password and email a login OTP
- `POST /auth/login/verify` — verify login OTP and issue access/refresh tokens
- `POST /auth/otp/resend` — resend a code after the cooldown

OTP security tests are in `backend/tests/test_otp_utils.py` and `backend/tests/test_auth_otp.py`.
