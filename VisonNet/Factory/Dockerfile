FROM ubuntu:20.10

RUN apt-get update
RUN apt-get install -y ffmpeg

WORKDIR /Factory

RUN mkdir MovFactory
RUN mkdir ImgFactory
COPY MovFactory/info.txt MovFactory/info.txt




#Non used
#RUN mkdir MovFactory/extracted
#RUN mkdir MovFactory/waiting

#COPY MovFactory/ffmpegCutNScaleMany.sh MovFactory/ffmpegCutNScaleMany.sh
#COPY MovFactory/info.txt MovFactory/info.txt
#COPY MovFactory/extracted/info.txt MovFactory/extracted/info.txt
#COPY MovFactory/waiting/info.txt MovFactory/waiting/info.txt



#RUN mkdir ImgFactory/compressed
#RUN mkdir ImgFactory/waiting

#COPY ImgFactory/ffmpegCompressImages.sh MovFactory/ffmpegCompressImages.sh


#Unused
#FROM alpine:latest #FROM alfg/ffmpeg:latest