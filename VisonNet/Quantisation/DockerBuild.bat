:: This is script which creates docker image used in the Quantisation of the model
:: Syntax = docker build -t name_of_the_new_image -f location_of_the_dockerfile location_of_the_context
docker build -t visonquantimage -f Dockerfile ..
pause