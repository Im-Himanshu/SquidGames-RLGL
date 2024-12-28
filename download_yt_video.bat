@echo off
:: Prompt user for YouTube URL
set /p youtube_url=Enter YouTube video URL: 

:: Extract video ID from the URL (assumes standard YouTube URL format)
for /f "tokens=2 delims==" %%A in ("%youtube_url%") do set video_id=%%A

:: Set output file path
set output_file=./download_videos/%video_id%.webm

:: Run yt-dlp command
yt-dlp "%youtube_url%" -o "%output_file%"

:: Notify user
if exist "%output_file%" (
    echo Video downloaded successfully: %output_file%
) else (
    echo Failed to download video.
)

pause
