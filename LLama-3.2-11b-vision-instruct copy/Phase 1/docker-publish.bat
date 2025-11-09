@echo off
REM Simple Docker Hub Publishing Script
REM Usage: docker-publish.bat <dockerhub-username>

set DOCKER_USERNAME=%1
if "%DOCKER_USERNAME%"=="" (
    echo Usage: docker-publish.bat YOUR_DOCKERHUB_USERNAME
    echo Example: docker-publish.bat johnsmith
    exit /b 1
)

set IMAGE_NAME=amex-offer-generator

echo Building Docker image...
docker build -t %IMAGE_NAME%:latest .

echo.
echo Tagging for Docker Hub...
docker tag %IMAGE_NAME%:latest %DOCKER_USERNAME%/%IMAGE_NAME%:latest

echo.
echo Logging into Docker Hub...
docker login

echo.
echo Pushing to Docker Hub...
docker push %DOCKER_USERNAME%/%IMAGE_NAME%:latest

echo.
echo Done! Your image is now available at:
echo docker pull %DOCKER_USERNAME%/%IMAGE_NAME%:latest
