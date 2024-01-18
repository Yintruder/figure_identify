#!/bin/bash

cd build
make clean
cmake ..
make -j4
cd ..
./CodeRecognizerExample