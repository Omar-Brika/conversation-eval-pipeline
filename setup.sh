# Virtual environment activation and requirements setup 
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "Creating virtual environment"
  python3 -m venv .venv
  source .venv/bin/activate
fi

if [ -f "requirements.txt" ]; then
  echo "Installing python dependencies"
  pip install -r requirements.txt
fi

