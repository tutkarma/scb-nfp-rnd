FROM python:3.9 as builder

RUN python3.9 -m venv /usr/share/python3/app
ADD requirements.txt /tmp/
RUN pip install --no-cache-dir -Ur /tmp/requirements.txt

FROM builder as app

ADD . /mnt/

WORKDIR /mnt

CMD ["python3", "app.py"]