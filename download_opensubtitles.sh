#!/bin/bash

echo "Download and extract OpenSubtitles 2018 en-es parallel data"
echo "to opensubtitles2018"
wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-pt.txt.zip -O temp.zip;
unzip temp.zip -d opensubtitles2018/;
rm temp.zip
