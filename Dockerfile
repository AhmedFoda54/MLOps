FROM python:3.9.19

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=600 -r requirements.txt

COPY Assignment1_A.py .

EXPOSE 8000

CMD ["python", "Assignment1_A.py"]