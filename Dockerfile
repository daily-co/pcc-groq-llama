FROM dailyco/pipecat-base:latest

COPY ./pipecat pipecat
COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./runner.py runner.py
COPY ./bot.py bot.py
