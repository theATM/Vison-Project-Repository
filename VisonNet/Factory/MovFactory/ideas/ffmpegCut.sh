#!/bin/bash
#First some variables:
INPUT_DIR="waiting" #NOT YET
OUTPUT_FOLDER="extracted"
echo "This script creates jpgs from mp4"
echo "Enter video file"
read IMPUT_FILE
IMPUT_FILE_NAME=$(basename -- "$IMPUT_FILE")
IMPUT_FILE_TYPE="${IMPUT_FILE_NAME##*.}"
IMPUT_FILE_NAME="${IMPUT_FILE_NAME%.*}"

mkdir "${OUTPUT_FOLDER}/${IMPUT_FILE_NAME}"
OUTPUT_FILE_PATH="./${OUTPUT_FOLDER}/${IMPUT_FILE_NAME}/${IMPUT_FILE_NAME}_%003d.png"
#Remember to set frames(-r) per second(-t) 
ffmpeg -ss 00:00 -i "${INPUT_DIR}/${IMPUT_FILE}" -r 30 $OUTPUT_FILE_PATH
read LINE