import musicbrainzngs

def get_music_infos(search_string):
    print("Searching song in musicbrainz...")
    #https://python-musicbrainzngs.readthedocs.io/en/v0.7.1/usage/#searching

    musicbrainzngs.set_useragent("UltraSinger", "0.1", "https://github.com/rakuri255/UltraSinger")

    artists = musicbrainzngs.search_artists(search_string)
    artist = artists['artist-list'][0]['name']
    release_groups = musicbrainzngs.search_release_groups(search_string, artist=artist)

    release = release_groups['release-group-list'][0]
    print(f"title: {release['title']}")
    print(f"artist: {artist}")
    print(f"artist-credit-phrase: {release['artist-credit-phrase']}")
    print(f"first-release-date: {release['first-release-date']}")
    genre = []
    for tag in release['tag-list']:
        genre.append(tag['name'])
    print(f"genre: {genre}")

    #release_id = release['id']
    #data = musicbrainzngs.get_image_list(release_id)
