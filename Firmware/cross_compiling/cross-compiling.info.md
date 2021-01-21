# Cross-compiling environment for Raspberry Pi

## Requirements 

- Docker 
- Visual Studio Code 
- raspberry pi docker image 

## Setup Process 

- Down load docker from their official website at: <https://www.docker.com/products/docker-desktop>.
- After downloading docker, you should Docker CLI available. Type docker --version to confirm successful installation.
- Pull down the Pi docker image from docker repo. Type 'docker pull mitchallen/pi-cross-compile'.
- Install docker extension on Visual Studio Code.

## Spinning up a container 

- Run the command 'docker run -it -d --entrypoint=/bin/bash -v <absolute path to firmware folder>:/Firmware mitchallen/pi-cross-compile'
- You can double check that your container is running by typing 'docker ps'
- Open Visual Studio and click on the Docker icon on the left hand side(whale icon)
- You should see a list of all your running containers on a new panel.
- Right click on the container you are going to use to develop.
- Click Attach Visual Studio.
- You are done! You can add files/folders through Visual Studio Code. All changes will be visible on Host OS because you mounted the Firmware folder to your container!

## Stopping/Reusing containers 
- To stop a container use the following command 'docker container stop <container id>.
    - To get the container id type 'docker ps'. The id is the number under the ID property.
-To restart a stopped container type 'docker container start <container id>.
    - To get container id type 'docker ps -a'
    - Notice the -a option.
