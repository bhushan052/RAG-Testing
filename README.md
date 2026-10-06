.venv\Scripts\Activate.ps1
python -m venv .venv

python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000