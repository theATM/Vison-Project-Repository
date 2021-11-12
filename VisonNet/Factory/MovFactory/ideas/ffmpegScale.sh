#!/bin/bash
#First some variables:
SAVE_DIR="scaled"
INPUT_DIR="waiting" #NOT YET
echo "This script scales video to 640:480"
echo "Enter video file"
read IMPUT_FILE
IMPUT_FILE_NAME=$(basename -- "$IMPUT_FILE")
IMPUT_FILE_TYPE="${IMPUT_FILE_NAME##*.}"
IMPUT_FILE_NAME="${IMPUT_FILE_NAME%.*}"

ffmpeg -i "${INPUT_DIR}/${IMPUT_FILE}" -vf scale=640:480 "${SAVE_DIR}/${IMPUT_FILE_NAME}s.${IMPUT_FILE_TYPE}"
read LINE