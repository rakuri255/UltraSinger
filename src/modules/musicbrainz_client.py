import musicbrainzngs
import string
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


def get_music_infos(search_string: str) -> tuple[str, str, str, str]:
    print(f"{ULTRASINGER_HEAD} Searching song in {blue_highlighted('musicbrainz')}")
    # https://python-musicbrainzngs.readthedocs.io/en/v0.7.1/usage/#searching

    musicbrainzngs.set_useragent("UltraSinger", "0.1", "https://github.com/rakuri255/UltraSinger")

    # search for artist and titel to get release on the first place
    artists = musicbrainzngs.search_artists(search_string)
    artist = artists['artist-list'][0]['name'].strip()
    release_groups = musicbrainzngs.search_release_groups(search_string, artist=artist)
    release = release_groups['release-group-list'][0]

    if 'artist-credit-phrase' in release:
        artist = release['artist-credit-phrase'].strip()

    title = None
    if 'title' in release:
        clean_search_string = search_string.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        clean_release_title = release['title'].translate(str.maketrans('', '', string.punctuation)).lower().strip()
        clean_artist = artist.translate(str.maketrans('', '', string.punctuation)).lower().strip()

        # prepare search string when title and artist are the same
        if clean_release_title == clean_artist:
            # remove the first appearance of the artist
            clean_search_string = clean_search_string.replace(clean_artist, "", 1)

        if clean_release_title in clean_search_string:
            title = release['title'].strip()
        else:
            print(
                f"{ULTRASINGER_HEAD} cant find title {red_highlighted(clean_release_title)} in {red_highlighted(clean_search_string)}")

    if title is None or artist is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No match found')}")
        return None, None, None, None

    print(f"{ULTRASINGER_HEAD} Found Titel and Artist {blue_highlighted(title)} by {blue_highlighted(artist)}")

    year = None
    if 'first-release-date' in release:
        year = release['first-release-date'].strip()
        print(f"{ULTRASINGER_HEAD} Found release year: {blue_highlighted(year)}")

    genres = None
    if 'tag-list' in release:
        genres = ""
        for tag in release['tag-list']:
            genres += f"{tag['name'].strip()},"
        print(f"{ULTRASINGER_HEAD} Found genres: {blue_highlighted(genres)}")

    return title, artist, year, genres
