FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m abhishek

RUN chown -R abhishek:abhishek /home/abhishek/

COPY --chown=abhishek . /home/abhishek/app/

USER abhishek

RUN cd /home/abhishek/app/ && pip3 install -r requirements.txt

WORKDIR /home/abhishek/app
