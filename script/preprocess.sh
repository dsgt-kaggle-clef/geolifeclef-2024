#!/bin/sh

#script that preprocess all the tif files to tiles.

set -e

RAW_DIR="/mnt/data/download"
PROC_DIR="/mnt/data/processed"

if [ "$#" -ge 2 ]; then
  PROC_DIR="$1"
  DATASET_DIR="$2"
else
  PROC_DIR="$PROC_DIR"
  DATA_SET_DIR="$RAW_DIR"
fi

mkdir -p "$PROC_DIR"

BLOCKXSIZE=256
BLOCKYSIZE=256

if [ "$#" -ge 4 ]; then
  BLOCKXSIZE="$3"
  BLOCKYSIZE="$4"
fi

echo "Directory for preprocess: $DATASET_DIR"
echo "BLOCKXSIZE: $BLOCKXSIZE"
echo "BLOCKYSIZE: $BLOCKYSIZE"

sudo chmod -R 777 "$DATASET_DIR"
echo "Permissions set for $DATASET_DIR."

find "$DATASET_DIR" -type f -name "*.tif" | while read file
do
  echo "Processing $file"
  output_file="${file##*/}"
  output_file="${PROC_DIR%/}/${output_file%.tif}_tiled.tif"
  gdal_translate -of GTiff -co "TILED=YES" -co "BLOCKXSIZE=$BLOCKXSIZE" -co "BLOCKYSIZE=$BLOCKYSIZE" "$file" "$output_file"
done

echo "Preprocess Completed"