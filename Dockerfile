FROM python:3.9-slim

WORKDIR /app

COPY model.pkl app.py requirements.txt ./
COPY templates/ templates/

RUN pip install -r requirements.txt

EXPOSE 5050

CMD ["python", "app.py"]
