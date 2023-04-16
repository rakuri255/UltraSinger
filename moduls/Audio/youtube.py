import yt_dlp


def get_youtube_title(url):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url,
            download=False  # We just want to extract the info
        )

    return result['title']


def download_youtube_audio(url, clear_filename, output_path):
    print("[UltraSinger] Downloading Audio")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '/' + clear_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))


def download_youtube_video(url, clear_filename, output_path):
    print("[UltraSinger] Downloading Video")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path + '/' + clear_filename + '.mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))
