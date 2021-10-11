FROM python:3.7-slim-buster

RUN mkdir /home/app

WORKDIR /home/app

COPY Pipfile .
COPY Pipfile.lock .

RUN apt-get update \
 && pip install --no-cache-dir pipenv pylint \
 && pipenv install --system --deploy --ignore-pipfile

COPY . .

ENTRYPOINT ["sh", "entrypoint.sh"]
