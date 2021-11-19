#!/bin/bash
#First some variables:
SAVE_DIR="compressed"
INPUT_DIR="waiting" #NOT YET
echo "This script compresses video"
echo "Enter video file"
read IMPUT_FILE
IMPUT_FILE_NAME=$(basename -- "$IMPUT_FILE")
IMPUT_FILE_TYPE="${IMPUT_FILE_NAME##*.}"
IMPUT_FILE_NAME="${IMPUT_FILE_NAME%.*}"

ffmpeg -i "${INPUT_DIR}/${IMPUT_FILE}" -vcodec libx265 -crf 30 "${SAVE_DIR}/${IMPUT_FILE_NAME}c.${IMPUT_FILE_TYPE}"
read LINE