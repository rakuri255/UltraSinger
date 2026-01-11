import musicbrainzngs
import string
import time
from urllib.error import URLError
from Levenshtein import ratio
from dataclasses import dataclass
from typing import Optional
from Settings import Settings

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


@dataclass
class SongInfo:
    title: str
    artist: str
    year: Optional[str] = None
    genres: Optional[str] = None
    cover_image_data: Optional[bytes] = None
    cover_url: Optional[str] = None


title_filter = [
    "official video",
    "official music video",
    "Offizielles Musikvideo",
]


MAX_RETRIES = 3


def __clean_string(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()


def __musicbrainz_request(func):
    for i in range(MAX_RETRIES):
        try:
            return func()
        except musicbrainzngs.musicbrainz.NetworkError:
            time.sleep(1)
    return None


def search_musicbrainz(title: str, artist) -> SongInfo:
    # Musicbrainz API documentation
    # https://python-musicbrainzngs.readthedocs.io/en/latest/api/

    musicbrainzngs.set_useragent("UltraSinger", Settings.APP_VERSION, "https://github.com/rakuri255/UltraSinger")

    # remove from search_string "official video"
    # todo: do we need filter?
    origin_title = title
    for filter in title_filter:
        title = title.lower().replace(filter.lower(), "").strip()
        if artist is not None:
            artist = artist.lower().replace(filter.lower(), "").strip()

    if artist is None:
        recording = __single_line_search(title)
    else:
        recording = __multi_line_search(artist, title)

    if recording is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No match found')}")
        return SongInfo(title=origin_title, artist="Unknown Artist")

    artist = recording['artist-credit-phrase']
    title = recording['title']
    print(
        f"{ULTRASINGER_HEAD} Found data on Musicbrainz: Artist={blue_highlighted(artist)} Title={blue_highlighted(title)}")

    year = __get_year(recording)
    genres = __get_genres(recording)
    image_data, image_url = __get_image(recording)

    return SongInfo(title=title, artist=artist, year=year, genres=genres, cover_image_data=image_data, cover_url=image_url)


def __single_line_search(search_string):
    search_string = __clean_string(search_string)
    search_string = __filter_words(search_string)

    artists = __musicbrainz_request(lambda: musicbrainzngs.search_artists(search_string, limit=10, artist=search_string))
    recordings = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(search_string, limit=100, artistname=search_string))

    if artists is None or recordings is None:
        return None

    found_artist = None

    for record in recordings['recording-list']:
        if found_artist is not None:
            break

        for artist_credit in record['artist-credit']:
            if found_artist is not None:
                break
            if isinstance(artist_credit, str):
                continue
            # todo: there is also an "alias-list". Maybe search also there?

            for artist in artists['artist-list']:
                if artist_credit['artist'] and artist_credit['artist']['id'] == artist['id']:
                    found_artist = record['artist-credit-phrase']
                    break

    if found_artist is None:
        return None

    recordings = [x for x in recordings['recording-list'] if
                  __clean_string(x['artist-credit-phrase']) == __clean_string(found_artist)]

    recording = None

    for record in recordings:
        if __clean_string(record['title']) in __clean_string(search_string):
            recording = record

    return recording


def __filter_words(search_string):
    for filter in title_filter:
        search_string = search_string.lower().replace(filter.lower(), "").strip()
    return search_string


def __multi_line_search(artist: str, title: str):
    # Try both combinations since we don't know which one is the artist and which one is the title
    artist1, title1 = artist, title
    artist2, title2 = title, artist

    result1 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(recording=title1, limit=10, artist=artist1, artistname=artist1))
    result2 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(recording=title2, limit=10, artist=artist2, artistname=artist2))

    if result1 is None:
        result1 = {'recording-count': 0, 'recording-list': []}
    if result2 is None:
        result2 = {'recording-count': 0, 'recording-list': []}

    # Filter result to ['artist-credit-phrase'] == artist
    record1 = [x for x in result1['recording-list'] if
               __clean_string(x['artist-credit-phrase']) == __clean_string(artist1)]
    record2 = [x for x in result2['recording-list'] if
               __clean_string(x['artist-credit-phrase']) == __clean_string(artist2)]

    if len(record1) > 0 and len(record2) > 0:
        best_match1 = max(record1, key=lambda x: ratio(__clean_string(x['title']), __clean_string(title1)))
        best_match2 = max(record2, key=lambda x: ratio(__clean_string(x['title']), __clean_string(title2)))

        is_match1 = ratio(__clean_string(title1), __clean_string(best_match1['title'])) > ratio(__clean_string(title2),
                                                                                                __clean_string(
                                                                                                    best_match2[
                                                                                                        'title']))

        if is_match1:
            recording = record1[0]
        else:
            recording = record2[0]

    elif len(record1) > 0:
        recording = record1[0]
    elif len(record2) > 0:
        recording = record2[0]
    elif result1['recording-count'] > 0:  # Artist = Title
        recording = result1['recording-list'][0]
    elif result2['recording-count'] > 0:  # Artist = Title
        recording = result2['recording-list'][0]
    else:
        recording = None

    return recording


def __get_image(recording) -> (bytes, str):
    image_data = None
    image_url = None
    if 'release-list' in recording:
        for release in recording['release-list']:
            try:
                image_data = __musicbrainz_request(lambda: musicbrainzngs.get_image_front(release['id']))
                if image_data is None:
                    continue

                image_list = __musicbrainz_request(lambda: musicbrainzngs.get_image_list(release['id']))
                if image_list is None:
                    continue

                for image in image_list['images']:
                    if image['front']:
                        image_url = image['image']
                        break
                break
            except musicbrainzngs.ResponseError:
                continue
    if image_data is not None:
        print(f"{ULTRASINGER_HEAD} Found cover image")

    return image_data, image_url


def __get_year(recording):
    year = None

    if 'release-list' not in recording:
        return year

    release_group_id = recording['release-list'][0]['release-group']['id']
    release_group = __musicbrainz_request(lambda: musicbrainzngs.get_release_group_by_id(release_group_id))

    if release_group is None:
        return year

    if 'first-release-date' not in release_group['release-group']:
        return year

    year = release_group['release-group']['first-release-date'].strip()
    year = year.split('-')[0]

    if year is not None:
        print(f"{ULTRASINGER_HEAD} Found year: {blue_highlighted(year)}")

    return year


def __get_genres(recording) -> str:
    # todo secondary-type-list ??
    genres = None
    if 'tag-list' in recording:
        genres = ""
        for tag in recording['tag-list']:
            genres += f"{tag['name'].strip()},"
    if genres is not None:
        print(f"{ULTRASINGER_HEAD} Found genres: {blue_highlighted(genres)}")
    return genres
