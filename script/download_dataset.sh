#!/bin/bash

# Script that downloads and extracts a dataset from GCS
# The dataset and destination can be configured

# Usage:
# ./download_extract_dataset.sh [DATASET_URL] [DESTINATION_DIR]
#
# Example:
# ./download_extract_dataset.sh gs://dsgt-clef-geolifeclef-2024/data/raw/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010 /mnt/data
#
# This will download the dataset from the specified URL and extract it to the specified destination directory.
# If no arguments are provided, default values are used for both the dataset URL and destination directory.

set -e

DEFAULT_DATASET_URL="gs://dsgt-clef-geolifeclef-2024/data/raw/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010"
DEFAULT_DEST_DIR="/mnt/data/download"

if [ "$#" -ge 2 ]; then
    DATASET_URL="$1"
    DESTINATION_DIR="$2"
else
    DATASET_URL="$DEFAULT_DATASET_URL"
    DESTINATION_DIR="$DEFAULT_DEST_DIR"
fi

sudo mkdir -p "$DESTINATION_DIR"

echo "Using dataset url: $DATASET_URL"
echo "Downloading the dataset to: $DESTINATION_DIR"

# might need it if we load raw zip file instead
sudo chmod -R 777 "$DESTINATION_DIR"
echo "Permissions set for $DESTINATION_DIR."

DATASET_URL="${DATASET_URL%/}/"
DATASET_URL_WILDCARD="${DATASET_URL}*"

gsutil -m cp -r "$DATASET_URL_WILDCARD" "$DESTINATION_DIR"

ls "$DESTINATION_DIR"
echo "Download Completed"