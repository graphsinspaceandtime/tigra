#!/bin/bash

command -v curl >/dev/null 2>&1 || { echo >&2 "curl required. please install it!"; exit 1; }

NETREP_URLS=(
  http://nrvis.com/download/data/misc/radoslaw_email.zip
)

NETREP_DIRS=(
  radoslaw_email
)

NETREP_CHECK_FILES=(
  radoslaw_email.edges
)

NETREP_CHECKSUMS=(
  cbdc821eb8949cfa36fb71b14aa3cf745e977de1fd2de0ece099c113df364969
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

SNAP_CHECKSUMS=(
  e00ba2415373dee52c00616065bcceaa4750e78de60d1855c76470600f10740f
  1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f
  1a416ccaf5a679996156dd31290dfcca3e7dde3431171afed1b1fd616fbd31e6
  2bf52f784cdba69730f912633aa1414b381059406ded2c38bc37658ead5b0773
  22a717fd1763199c95b10dbea80af4fb658f0b9e71cd9eb8a1fff51462238cdc
  d99a66429d0cb69b6f61be71cac3064868cd683b251140b232f377bf42112be7
  d45012e758e26bc28f6b610f1146aee0e87728a79e69cc6d696dd180f19dff66
)

for i in ${!NETREP_URLS[@]}
do
  echo "${NETREP_CHECKSUMS[$i]}  netrep/${NETREP_DIRS[$i]}/${NETREP_CHECK_FILES[$i]}" | sha256sum --quiet --check >/dev/null 2>&1
  if [ $? -eq 1 ]
  then
    echo "Checksum incorrect or file not found for ${NETREP_DIRS[$i]}"
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
  echo "${SNAP_CHECKSUMS[$i]}  snap/${SNAP_FILENAMES[$i]%\.gz}" | sha256sum --quiet --check
  if [ $? -eq 1 ]
  then
    echo "Checksum incorrect or file not found for ${SNAP_FILENAMES[$i]}"
    echo "Downloading ${SNAP_FILENAMES[$i]%\.gz}"
    rm "snap/${SNAP_FILENAMES[$i]}"
    curl --progress-bar --output "snap/${SNAP_FILENAMES[$i]}" "${SNAP_URLS[$i]}"
    gzip --decompress "snap/${SNAP_FILENAMES[$i]}"
  else
    echo "Skipping ${SNAP_FILENAMES[$i]%\.gz}... already exists"
  fi
done
