# syntax=docker/dockerfile:experimental

#This is Dockerfile creating the linux container (linux is needed to quantize model)
#The context (defined during build) mush have all those "COPY"ed files "inside" or it will not build
#Docker interpreter (opiton in some IDEs) can be used to use and debug code


#get Basics
FROM ubuntu:20.10
#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl

FROM python:3.7.12


WORKDIR /Quant


COPY Environment/quantenv/QuantisationRequirements.txt qr.txt

RUN --mount=type=cache,target=/root/.cache \
	pip install -r qr.txt


RUN mkdir Models
RUN mkdir Models/Quantin
COPY Models/Quantin/* Models/Quantin
RUN mkdir Models/Quantout

RUN mkdir Data
RUN mkdir Data/dataset
RUN mkdir Data/testset
RUN mkdir Data/valset

RUN mkdir Network
COPY Network/parameters.py Network/parameters.py 

RUN mkdir Network/Bank
COPY Network/Bank/bankset.py Network/Bank/bankset.py 
COPY Network/Bank/banksethelpers.py Network/Bank/banksethelpers.py 
COPY Network/Bank/banksetuttils.py Network/Bank/banksetuttils.py
COPY Network/Bank/transforms.py Network/Bank/transforms.py

RUN mkdir Network/Architecture
COPY Network/Architecture/model.py Network/Architecture/model.py 
COPY Network/Architecture/modeltype.py Network/Architecture/modeltype.py 
COPY Network/Architecture/mobilenet.py Network/Architecture/mobilenet.py 


RUN mkdir Network/Functional
COPY Network/Functional/evaluate.py Network/Functional/evaluate.py 
COPY Network/Functional/staticquant.py Network/Functional/staticquant.py



COPY ".idea" ".idea"


