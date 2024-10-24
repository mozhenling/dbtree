#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'update links to the master branch'

git push origin master

echo '------- update complete --------'