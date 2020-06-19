import os
import json
import random

def generate_7digital_tracklist():
    dir = os.path.abspath(os.getcwd()) + '/MSD_previews'
    tracklist = []
    with open('7digital_tracklist.txt', 'w') as tracklist_file:
        for root, dirs, files in os.walk(dir):
            for name in files:
                file = os.path.join(root, name)
                if name.endswith('.mp3') and not name.startswith('._'):
                    track_id = name.split('.')[0]
                    print(track_id)
                    tracklist_file.write(track_id+'\n')

def retrieve_7digital_tracklist():
    tracklist = []
    with open('7digital_tracklist.txt', 'r') as tracklist_file:
        lines = tracklist_file.readlines()
        for line in lines:
            track_id = line.replace('\n','')
            tracklist.append(track_id)
    return tracklist

#generate_7digital_tracklist()

def mapping():
    mapping = {}
    
    with open('unique_tracks.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            track_id, song_id, artist, title = line.split('<SEP>')
            print('{} is {}'.format(track_id,song_id))
            mapping[song_id] = track_id

    return mapping

def getting_all_tracks():
    tracks = []
    n = 0
    # File containing the list of 1 million tracks (track id // song id // artist name // song name)
    with open('unique_tracks.txt', 'r') as file:
        lines = file.readlines()
        print('total tracks: {}'.format(len(lines)))
        for line in lines:
            track_id, song_id, artist, title = line.split('<SEP>')
            n += 1
            print('appending track {} / {}'.format(track_id,n))
            tracks.append(track_id)

    # File containing the list of 1 million track ids
    with open('all_tracks.txt', 'w') as file:
        for track_id in tracks:
            print('writing track {}'.format(track_id))
            file.write(track_id + '\n')

def train_test_split():
    train_tracks = []
    test_tracks = []
    ratio = 0.7

    #FILE = 'all_sample_subset_tracks.txt'
    FILE = 'all_tracks_100users_50songs.txt'
    with open(FILE, 'r') as file:
        lines = file.readlines()
        for line in lines:
            track_id = line
            value = random.uniform(0, 1)
            print(value)
            if value <= ratio:
                train_tracks.append(track_id)
            else:
                test_tracks.append(track_id)

    #TRAIN_OUTPUT_FILE = 'sample_train_tracks_7digital.txt'
    TRAIN_OUTPUT_FILE = 'train_100users_50songs.txt'
    with open(TRAIN_OUTPUT_FILE, 'w') as file:
        for track_id in train_tracks:
            file.write(track_id)

    #TEST_OUTPUT_FILE = 'sample_test_tracks_7digital.txt'
    TEST_OUTPUT_FILE = 'test_100users_50songs.txt'
    with open(TEST_OUTPUT_FILE, 'w') as file:
        for track_id in test_tracks:
            file.write(track_id)

    print('TOTAL TRAIN TRACKS: {}'.format(len(train_tracks)))
    print('TOTAL TEST TRACKS: {}'.format(len(test_tracks)))

def match_to_7digital(mapping):
    songs = {}
    new_file = []

    users = 0
    prev_user = None
    with open('train_triplets.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            new_line = ""
            user, song, play_count = line.replace('\n','').split('\t')
            print('mapping {}'.format(song))
            if song in songs:
                print('{} already matched with {}'.format(song, songs[song]))
                new_line = user + '\t' + songs[song] + '\t' + play_count + '\n'
            else:
                songs[song] = mapping[song]
                print('{} matched with {}'.format(mapping[song],song))
                new_line = user + '\t' + mapping[song] + '\t' + play_count + '\n'
            new_file.append(new_line)
            if user != prev_user:
                users += 1
            if users < 0:
                break
            prev_user = user

    with open('train_triplets_7digital.txt', 'w') as triplets_file:
        for line in new_file:
            triplets_file.write(line)

def fix_all_subset_tracks():
    with open('all_subset_tracks.txt', 'r') as file:
        lines = file.readlines()
        new_lines = []
        for line in lines:
            n = 8
            new_lines = [line[i:i+n] for i in range(0, len(line), n)]
    
    with open('all_subset_tracks.txt', 'w') as file:
        for line in new_lines:
            file.write(line + '\n')

def match():
    songs = []

    with open('sample_train_triplets_7digital.txt', 'r') as file:
        print('reading lines...')
        lines = file.readlines()
        i = 0
        for line in lines:
            new_line = ""
            user, song, play_count = line.replace('\n','').split('\t')
            print('mapping {} {}'.format(song,i))
            if not song in songs:
                songs.append(song)
            else:
                print('else')
            i += 1

    with open('all_sample_subset_tracks.txt', 'w') as file:
        for song in songs:
            file.write(song + '\n')

def generate_all_tracks_list():
    songs = []

    with open('train_triplets_7digital_100users_50songs.txt', 'r') as file:
        print('reading lines...')
        lines = file.readlines()
        i = 0
        for line in lines:
            new_line = ""
            user, song, play_count = line.replace('\n','').split('\t')
            print('mapping {} {}'.format(song,i))
            if not song in songs:
                songs.append(song)
            else:
                print('else')
            i += 1

    print('Total tracks: {}'.format(len(songs)))

    with open('all_tracks_100users_50songs.txt', 'w') as file:
        for song in songs:
            file.write(song + '\n')

def audio_exists(track_id):
    audio_path = 'msd/audio/{}/{}/{}/{}.mp3'.format(track_id[2],track_id[3],track_id[4],track_id)
    return os.path.isfile(audio_path)

def generate_filtered_triplets(max_users=100,min_songs=50):
    new_lines = []

    prev_user_id = ""
    provisional_lines = []
    user_n_songs = 0
    user_not_found_songs = 0
    filtered_users = []
    n_users = 0
    total_users = 0

    with open('train_triplets_7digital.txt', 'r') as file:
    #with open('sample_train_triplets_7digital.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            user, song, play_count = line.replace('\n','').split('\t')

            if prev_user_id and user != prev_user_id:
                total_users += 1
                print('User: {}'.format(prev_user_id))
                print('Songs found: {}/{}'.format(user_n_songs, user_n_songs+user_not_found_songs))
                if user_n_songs >= min_songs:
                    filtered_users.append(prev_user_id)
                    n_users += 1
                    for line in provisional_lines:
                        new_lines.append(line)
                    if n_users == max_users:
                        break
                provisional_lines = []
                if audio_exists(song):
                    user_n_songs = 1
                    provisional_lines.append(line)
                else:
                    #print("{} doesn't exist".format(song))
                    user_n_songs = 0
                    user_not_found_songs = 1 
            else:
                if audio_exists(song):
                    provisional_lines.append(line)
                    user_n_songs += 1
                else:
                    #print("{} doesn't exist".format(song))
                    user_not_found_songs += 1 
            
            prev_user_id = user

    with open('train_triplets_7digital_{}users_{}songs.txt'.format(n_users,min_songs), 'w') as file:
        for line in new_lines:
            file.write(line)

    print(filtered_users)
    #print(new_lines)
    print('Total filtered users: {}'.format(len(filtered_users)))
    print('Total users: {}'.format(total_users))

#match(mapping())
#fix_all_subset_tracks()
#train_test_split()
#match()

#generate_filtered_triplets()
#print(audio_exists('TRPRWYP128F146DE1D'))
#generate_all_tracks_list()
train_test_split()