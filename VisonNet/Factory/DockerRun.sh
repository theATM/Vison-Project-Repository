#!/bin/bash
# For Linux
docker run -it --rm -v ${PWD}/MovFactory:/Factory/MovFactory -v ${PWD}/ImgFactory:/Factory/ImgFactory  --name visondatafactorycontainer visondatafactoryimage

#-v ${PWD}/MovFactory/waiting:/Factory/MovFactory/waiting  -v ${PWD}/MovFactory/extracted:/Factory/MovFactory/extracted -v ${PWD}/MovFactory/compressed:/Factory/MovFactory/compressed -v ${PWD}/MovFactory/waiting:/Factory/MovFactory/waiting