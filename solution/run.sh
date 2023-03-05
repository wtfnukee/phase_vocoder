#!/bin/bash

input_file=$1
output_file=$2
time_stretch_ratio=$3

pip install -r requirements.txt
python vocoder.py $input_file $output_file $time_stretch_ratio