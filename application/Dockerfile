FROM python:3.8.10

RUN apt-get update; apt-get clean
# Install wget.
RUN apt-get install -y wget
RUN apt-get install -y gnupg
# Set the Chrome repo.
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
# Install Chrome.
RUN apt-get update && apt-get -y install google-chrome-stable



WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
