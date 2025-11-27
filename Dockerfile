FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 解压知识库
RUN unzip knowledge.zip -d /code/knowledge && rm knowledge.zip

EXPOSE 8501
CMD ["streamlit", "run", "legal_rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
