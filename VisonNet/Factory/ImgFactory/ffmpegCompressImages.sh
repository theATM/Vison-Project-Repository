#!/bin/bash
#this script cuts video into frames - images (if video is in 30fps) and resizes them to 640:480p
#To make it work:
#Script must be (in default case) inside this specific structure:
# Main Dir
# L this script (ffmpegCutNScaleMany.sh)
# L extracted - there will be our images
# L.. <videodir> - there will be images from specific video
# L waiting - file where we copy videos we want to cut

#First some variables:
INPUT_DIR="waiting"
OUTPUT_FOLDER="compressed"
echo "This script creates scaled img from img"
echo "They are also scaled to 244:244"




scale_image ()
{
	#Variables
	image_full_path=$1  							# dir1/dir2/image_name.type
	whole_path=$2									# dir1/dir2/
	image_file=$(basename -- "$image_full_path") 	# image_name.type
	image_type="${image_name##*.}" 					# .type
	image_name="${image_name%.*}" 					# image_name
	
	#Scale img to 244:244
	ffmpeg -i "${INPUT_DIR}/${video_file_path}" -vf scale=244:244 "${INPUT_DIR}/tmp.${video_file_type}"
}

get_all_files ()
{
	#Go through all files
	parent_dir=$1
	whole_path="{$2}{$parent_dir}/"
	for file_or_dir in $parent_dir/*
	do
		#echo "work"
		#echo $file_or_dir
		# If it is a file
		if [ -f "${file_or_dir}" ]; then 
			#echo "file"
			file=$file_or_dir
			scale_image $file
		# If this is a dir
		elif [ -d "${file_or_dir}" ]; then
			#echo "dir"
			dir=$file_or_dir
			get_all_files $dir $whole_path
		else
			echo "none"
		fi
	done
}	
	
		#video_file_path=$(echo $video | sed "s/^${INPUT_DIR}\///") 
		#video_file_name=$(basename -- "$video_file_path")
		#video_file_type="${video_file_name##*.}"
		#video_file_name="${video_file_name%.*}"
		#Scale video to 640:480p
		#ffmpeg -i "${INPUT_DIR}/${video_file_path}" -vf scale=244:244 "${INPUT_DIR}/tmp.${video_file_type}"		
		#Create dir to save images
		#mkdir "${OUTPUT_FOLDER}/${video_file_name}"
		#output_file_path="./${OUTPUT_FOLDER}/${video_file_name}/${video_file_name}s_%003d.png"
		#Remember to set frames(-r) per second(-t) 
		#Cut video to images:
		#ffmpeg -ss 00:00 -i "${INPUT_DIR}/tmp.${video_file_type}" -r 30 $output_file_path
		#rm "${INPUT_DIR}/tmp.${video_file_type}"	
		

get_all_files $INPUT_DIR
echo "Done"
#to hold terminal at the end:
read LINE