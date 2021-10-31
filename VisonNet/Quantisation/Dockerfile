# syntax=docker/dockerfile:experimental
#get Basics
FROM ubuntu:20.10
#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl

FROM python:3.6.14


WORKDIR /VisonApp

COPY Quantisation/QuantisationRequirements.txt qr.txt

RUN --mount=type=cache,target=/root/.cache \
	pip install -r qr.txt

COPY Models/* Models/*

RUN mkdir dataset
RUN mkdir testset
RUN mkdir valset

RUN mkdir Network
COPY Network/parameters.py Network/parameters.py 
COPY Network/bankset.py Network/bankset.py 
COPY Network/model.py Network/model.py 
COPY Network/modeltype.py Network/modeltype.py 
COPY Network/mobilenet.py Network/mobilenet.py 
COPY Network/evaluate.py Network/evaluate.py 
COPY Quantisation/quantize.py Quantisation/quantize.py
COPY Quantisation/staticquant.py Quantisation/staticquant.py
COPY ".idea" ".idea"


