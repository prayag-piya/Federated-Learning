FROM python:3.10

WORKDIR /app

COPY requriment.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir model
RUN mkdir configs

RUN python init_tokenizer.py
CMD ["python", "server.py", "server"]