#!/bin/bash
#this script cuts video into frames - images (if video is in 30fps) and resizes them to 244:244p
#To make it work:
#Script must be (in default case) inside this specific structure:
# Main Dir
# L this script (ffmpegCutNScaleMany.sh)
# L extracted - there will be our images
# L.. <videodir> - there will be images from specific video
# L waiting - file where we copy videos we want to cut
#When editing this file on Windows:
#After edit you need to change cloasing character
#Use "sed -i -e 's/\r$//' scriptname.sh" inside docker machine
#And then run the script

#First some variables:
INPUT_DIR="waiting"
OUTPUT_FOLDER="extracted"
echo "This script creates img from all mp4 in waiting folder"
echo "They are also scaled to 244:244p"


for video in $INPUT_DIR/*
do
	# If it is a file
	if [ -f "${video}" ]; then
		video_file_path=$(echo $video | sed "s/^${INPUT_DIR}\///") 
		video_file_name=$(basename -- "$video_file_path")
		video_file_type="${video_file_name##*.}"
		video_file_name="${video_file_name%.*}"
		#Scale video to 244:244
		ffmpeg -i "${INPUT_DIR}/${video_file_path}" -vf scale=244:244 "${INPUT_DIR}/tmp.${video_file_type}"		
		#Create dir to save images
		mkdir "${OUTPUT_FOLDER}/${video_file_name}"
		output_file_path="./${OUTPUT_FOLDER}/${video_file_name}/${video_file_name}s_%003d.png"
		#Remember to set frames(-r) per second(-t) 
		#Cut video to images:
		ffmpeg -ss 00:00 -i "${INPUT_DIR}/tmp.${video_file_type}" -r 30 $output_file_path
		rm "${INPUT_DIR}/tmp.${video_file_type}"	
		
	fi
done
echo "Done"
#to hold terminal at the end:
read LINE