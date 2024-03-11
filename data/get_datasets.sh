#!/bin/bash

command -v curl >/dev/null 2>&1 || { echo >&2 "curl required. please install it!"; exit 1; }

NETREP_URLS=(
  http://nrvis.com/download/data/misc/radoslaw_email.zip
)

NETREP_DIRS=(
  radoslaw_email
)

SNAP_URLS=(
  https://snap.stanford.edu/data/CollegeMsg.txt.gz
  https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
  https://snap.stanford.edu/data/sx-askubuntu.txt.gz
  https://snap.stanford.edu/data/sx-mathoverflow.txt.gz
  https://snap.stanford.edu/data/sx-stackoverflow.txt.gz
  https://snap.stanford.edu/data/sx-superuser.txt.gz
  https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz
)

SNAP_FILENAMES=(
  CollegeMsg.txt.gz
  email-Eu-core-temporal.txt.gz
  sx-askubuntu.txt.gz
  sx-mathoverflow.txt.gz
  sx-stackoverflow.txt.gz
  sx-superuser.txt.gz
  wiki-talk-temporal.txt.gz
)

for i in ${!NETREP_URLS[@]}
do
  if [ ! -d "netrep/${NETREP_DIRS[$i]}" ]
  then
    echo "Downloading ${NETREP_DIRS[$i]}"
    TEMP_FILENAME="$(mktemp)"
    curl --progress-bar --output "$TEMP_FILENAME" "${NETREP_URLS[$i]}"
    mkdir -p "netrep/${NETREP_DIRS[$i]}"
    unzip -q -d "$PWD/netrep/${NETREP_DIRS[$i]}" "$TEMP_FILENAME"
    rm "$TEMP_FILENAME"
  else
    echo "Skipping ${NETREP_DIRS[$i]}... already exists"
  fi
done

mkdir -p "snap"
for i in ${!SNAP_URLS[@]}
do
  if [ ! -f "snap/${SNAP_FILENAMES[$i]%\.gz}" ]
  then
    echo "Downloading ${SNAP_FILENAMES[$i]%\.gz}"
    curl --progress-bar --output "snap/${SNAP_FILENAMES[$i]}" "${SNAP_URLS[$i]}"
    gzip --decompress "snap/${SNAP_FILENAMES[$i]}"
  else
    echo "Skipping ${SNAP_FILENAMES[$i]%\.gz}... already exists"
  fi
done
