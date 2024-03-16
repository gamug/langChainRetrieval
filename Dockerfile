# FROM python:3.10 as requirements-stage
FROM python:3

# COPY --from=requirements-stage ./requirements.txt /code/requirements.txt

RUN python3 -m pip install certifi
RUN python3 -m pip install --upgrade pip


COPY . /code/

RUN apt-get update --allow-unauthenticated --allow-insecure-repositories -y

#installing requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN export SSL_CERT_FILE=$(pypy -c "import certifi; print(certifi.where())")
RUN echo 'export SSL_CERT_FILE="'$SSL_CERT_FILE'"' >> /code/.bashrc
RUN cd /code

EXPOSE 3001
WORKDIR /code

# Heroku uses PORT, Azure App Services uses WEBSITES_PORT, Fly.io uses 8080 by default
CMD ["/bin/bash", "-c", "source /code/.bashrc && uvicorn server.main:app --host 0.0.0.0 --port 3001"]