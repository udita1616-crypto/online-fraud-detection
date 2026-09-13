# Online Fraud Detection System

A prototype Flask application for online transaction fraud detection.

## What it does

The project creates its own synthetic historical transaction dataset at startup,
trains a Decision Tree on that historical data, and evaluates new transactions
using the project's predefined rules.

### Predefined rules

**Fraud**
- Amount > ₹150,000 AND transaction hour is between 11 PM and 5 AM AND transaction is international
- OR failed login attempts >= 4

**Suspicious**
- Failed login attempts >= 3
- OR amount > ₹50,000 AND transaction is international

**Safe**
- Everything else

The predefined rules are the final authority for the prototype result.

## Project structure

```text
online-fraud-detection/
├── app/
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt
python app/app.py
```

Then open:

http://127.0.0.1:5000/

## Important

This is a demonstration prototype. The historical dataset is synthetic and
there is no connection to a bank, payment gateway, or live transaction API.
