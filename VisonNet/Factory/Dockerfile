FROM ubuntu:20.10

RUN apt-get update
RUN apt-get install -y ffmpeg
RUN apt-get install nano

WORKDIR /Factory

RUN mkdir MovFactory
RUN mkdir ImgFactory
COPY info.txt info.txt

#Probably can be done easier with:
#FROM alfg/ffmpeg:latest
#Go see ffmpeg dockerhub page for more