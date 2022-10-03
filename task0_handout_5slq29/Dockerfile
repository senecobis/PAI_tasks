FROM --platform=linux/amd64 python:3.8-slim
ADD ./requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y gcc
RUN apt-get install libssl-dev

RUN pip install -U pip && pip install -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD python -u checker_client.py
