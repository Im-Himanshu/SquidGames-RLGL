@echo off
:: Prompt user for folder path
set /p folder_path="Enter the folder path containing MP4 videos (without quotes): "

:: Check if folder exists
if not exist "%folder_path%" (
    echo Folder does not exist. Exiting.
    pause >nul
    exit /b
)

:: Create a new folder for processed videos
set output_folder=%folder_path%\Processed_Videos
if not exist "%output_folder%" (
    mkdir "%output_folder%"
)

:: Enable delayed expansion for handling variables in loops
setlocal enabledelayedexpansion

:: Process each MP4 file in the folder
for %%F in ("%folder_path%\*.mp4") do (
    :: Extract file name without extension
    set "file_name=%%~nF"
    set "output_file=%output_folder%\!file_name!_extended.mp4"

    :: Run FFmpeg command to increase duration by replicating frames
    ffmpeg -i "%%F" -vf "setpts=3*PTS" -y "!output_file!"

    :: Notify user
    if exist "!output_file!" (
        echo Processed %%F to increase duration: !output_file!
    ) else (
        echo Failed to process %%F
    )
)

:: Exit message
echo Processing complete. Processed videos are in: %output_folder%
echo Press any key to exit.
pause >nul
