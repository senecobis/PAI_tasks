FROM python:3.8-slim
ADD ./requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install libssl-dev
RUN apt-get -y install gcc

RUN pip install -U pip && pip install -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
#ADD pytransform /code/task0/pytransform
WORKDIR /code
CMD python -u checker_client.py
