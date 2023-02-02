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
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': output_path + '/' + clear_filename + '.mp3',
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
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path + '/' + clear_filename + '.mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))
