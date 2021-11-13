#!/bin/bash
#this script cuts video into frames - images (if video is in 30fps) and resizes them to 640:480p
#To make it work:
#Script must be (in default case) inside this specific structure:
# Main Dir
# L this script (ffmpegCutNScaleMany.sh)
# L extracted - there will be our images
# L.. <videodir> - there will be images from specific video
# L waiting - file where we copy videos we want to cut
#Better to edit this script on linux - or from inside the docker container - using nano
#When editing this file on Windows:
#After edit you need to change cloasing character
#Use "sed -i -e 's/\r$//' scriptname.sh" inside docker machine
#And then run the script

#Into:
#First some variables:
INPUT_DIR="waiting"
OUTPUT_DIR="compressed"
#And Greetings
echo "This script creates scaled img from img"
echo "They are also scaled to 244:244"



#Actual Scaling:
scale_image ()
{
	#Variables
	image_full_path=$1  						# dir1/dir2/image_name.type
	full_dir_path=$2						# dir1/dir2/
	dir_path=${full_dir_path/$INPUT_DIR\//}				# dir2/
	image_file=$(basename -- "$image_full_path") 			# image_name.type
	echo $image_name
	image_type="${image_name##*.}" 					# .type
	echo $image_type
	image_name="${image_name%.*}" 					# image_name
	echo $image_name
	#Check if needed to add dirs
	if [ ! -z $dir_path ]; then
		mkdir "${OUTPUT_DIR}/${dir_path}"
	fi
	#Define out path
	full_out_path="${OUTPUT_DIR}/${dir_path}${image_name}c${image_type}"
	echo $image_name
	echo $image_full_path
	#Scale img to 244:244
	ffmpeg -i $image_full_path -vf scale=244:244 $full_out_path
}

#Main Loop:
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
	
#Start:
get_all_files $INPUT_DIR
echo "Done"
#to hold terminal at the end:
read LINE
