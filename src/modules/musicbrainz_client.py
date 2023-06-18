import musicbrainzngs
import string
from src.modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


def get_music_infos(search_string: str) -> tuple[str, str, str, []]:
    print(f"{ULTRASINGER_HEAD} Searching song in {blue_highlighted('musicbrainz')}")
    # https://python-musicbrainzngs.readthedocs.io/en/v0.7.1/usage/#searching

    musicbrainzngs.set_useragent("UltraSinger", "0.1", "https://github.com/rakuri255/UltraSinger")

    artists = musicbrainzngs.search_artists(search_string)
    artist = artists['artist-list'][0]['name']
    release_groups = musicbrainzngs.search_release_groups(search_string, artist=artist)
    release = release_groups['release-group-list'][0]

    title = None
    if 'title' in release:
        clean_search_string = search_string.translate(str.maketrans('', '', string.punctuation)).lower()
        clean_release_title = release['title'].translate(str.maketrans('', '', string.punctuation)).lower()
        if clean_release_title in clean_search_string:
            title = release['title']
        else:
            print(
                f"{ULTRASINGER_HEAD} cant find title {red_highlighted(release['title'])} in {red_highlighted(search_string)}")

    if 'artist-credit-phrase' in release:
        artist = release['artist-credit-phrase']

    if title is None or artist is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No match found')}")
        return None, None, None, []

    print(f"{ULTRASINGER_HEAD} Found and using {blue_highlighted(title)} by {blue_highlighted(artist)}")

    year = None
    if 'first-release-date' in release:
        year = release['first-release-date']
        print(f"{ULTRASINGER_HEAD} Release year: {blue_highlighted(year)}")

    genre = []
    if 'tag-list' in release:
        for tag in release['tag-list']:
            genre.append(tag['name'])

    return title, artist, year, genre
