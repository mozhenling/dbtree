#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add angular resampling'

git push origin master

echo '------- update complete --------'