@echo off
:: Prompt user for input
set /p start_time=Enter start time (ss): 
set /p duration=Enter duration (t): 
set /p input_file=Enter input file name (with extension):

:: Extract folder and file name without extension
for %%F in ("%input_file%") do set folder=%%~dpF
for %%F in ("%input_file%") do set file_name=%%~nF

:: Create a new folder for trimmed videos
set trimmed_folder=%folder%%file_name%_trimmed
if not exist "%trimmed_folder%" (
    mkdir "%trimmed_folder%"
)

:: Construct output file name
set output_file=%trimmed_folder%\trimmed-%start_time%-%duration%-%file_name%.mp4

:: Run FFmpeg command
ffmpeg -ss %start_time% -t %duration% -i "%input_file%" -y -c copy "%output_file%"

:: Notify user
if exist "%output_file%" (
    echo Trimmed video created successfully: %output_file%
) else (
    echo Failed to create trimmed video.
)

:: Exit message
echo Press any key to exit.
pause >nul
