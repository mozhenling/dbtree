#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'save dict to .json'

git push origin master

echo '------- update complete --------'