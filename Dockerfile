FROM nvcr.io/nvidia/pytorch:24.02-py3

# Update repositories
RUN apt update
RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN pip install nltk
