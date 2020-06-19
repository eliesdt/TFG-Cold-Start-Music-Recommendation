import os
import json
import numpy as np

import essentia
import essentia.standard as es

from datetime import datetime

from musicnn.tagger import top_tags
from musicnn2.musicnn.tagger import penultimate_features

TRAIN_FILE = 'train_100users_50songs.txt'
TEST_FILE = 'test_100users_50songs.txt'
TRACKLIST_FILE = 'all_tracks_100users_50songs.txt'
TRIPLETS_FILE = 'train_triplets_7digital_100users_50songs.txt'

PREVIEWS_FOLDER = 'msd'
TAGS_FOLDER = 'sample_tags'
PENULTIMATE_FOLDER = 'penultimate'
FEATURE_FILES_FOLDER = 'sample_features'

RECOMMENDATIONS_N = 50

FEATURE_METHOD = True
TAG_METHOD = False
PENULTIMATE_METHOD = False

NOT_AVAILABLE_SONGS = ['TRFWOAG128F14B12CB','TRPZALY128F4296B9F','TRXGPCA128F425135A']

FEATURES = ['lowlevel.average_loudness',
			'lowlevel.barkbands_crest.mean',
			'lowlevel.barkbands_crest.stdev',
			'lowlevel.barkbands_flatness_db.mean',
			'lowlevel.barkbands_flatness_db.stdev',
			'lowlevel.barkbands_kurtosis.mean',
			'lowlevel.barkbands_kurtosis.stdev',
			'lowlevel.barkbands_skewness.mean',
			'lowlevel.barkbands_skewness.stdev',
			'lowlevel.barkbands_spread.mean',
			'lowlevel.barkbands_spread.stdev',
			'lowlevel.dissonance.mean',
			'lowlevel.dissonance.stdev',
			'lowlevel.dynamic_complexity',
			'lowlevel.erbbands_crest.mean',
			'lowlevel.erbbands_crest.stdev',
			'lowlevel.erbbands_flatness_db.mean',
			'lowlevel.erbbands_flatness_db.stdev',
			'lowlevel.erbbands_kurtosis.mean',
			'lowlevel.erbbands_kurtosis.stdev',
			'lowlevel.erbbands_skewness.mean',
			'lowlevel.erbbands_skewness.stdev',
			'lowlevel.erbbands_spread.mean',
			'lowlevel.erbbands_spread.stdev',
			'lowlevel.hfc.mean',
			'lowlevel.hfc.stdev',
			'lowlevel.loudness_ebu128.integrated',
			'lowlevel.loudness_ebu128.loudness_range',
			'lowlevel.loudness_ebu128.momentary.mean',
			'lowlevel.loudness_ebu128.momentary.stdev',
			'lowlevel.loudness_ebu128.short_term.mean',
			'lowlevel.loudness_ebu128.short_term.stdev',
			'lowlevel.melbands_crest.mean',
			'lowlevel.melbands_crest.stdev',
			'lowlevel.melbands_flatness_db.mean',
			'lowlevel.melbands_flatness_db.stdev',
			'lowlevel.melbands_kurtosis.mean',
			'lowlevel.melbands_kurtosis.stdev',
			'lowlevel.melbands_skewness.mean',
			'lowlevel.melbands_skewness.stdev',
			'lowlevel.melbands_spread.mean',
			'lowlevel.melbands_spread.stdev',
			'lowlevel.pitch_salience.mean',
			'lowlevel.pitch_salience.stdev',
			'lowlevel.silence_rate_20dB.mean',
			'lowlevel.silence_rate_20dB.stdev',
			'lowlevel.silence_rate_30dB.mean',
			'lowlevel.silence_rate_30dB.stdev',
			'lowlevel.silence_rate_60dB.mean',
			'lowlevel.silence_rate_60dB.stdev',
			'lowlevel.spectral_centroid.mean',
			'lowlevel.spectral_centroid.stdev',
			'lowlevel.spectral_complexity.mean',
			'lowlevel.spectral_complexity.stdev',
			'lowlevel.spectral_decrease.mean',
			'lowlevel.spectral_decrease.stdev',
			'lowlevel.spectral_energy.mean',
			'lowlevel.spectral_energy.stdev',
			'lowlevel.spectral_energyband_high.mean',
			'lowlevel.spectral_energyband_high.stdev',
			'lowlevel.spectral_energyband_low.mean',
			'lowlevel.spectral_energyband_low.stdev',
			'lowlevel.spectral_energyband_middle_high.mean',
			'lowlevel.spectral_energyband_middle_high.stdev',
			'lowlevel.spectral_energyband_middle_low.mean',
			'lowlevel.spectral_energyband_middle_low.stdev',
			'lowlevel.spectral_entropy.mean',
			'lowlevel.spectral_entropy.stdev',
			'lowlevel.spectral_flux.mean',
			'lowlevel.spectral_flux.stdev',
			'lowlevel.spectral_kurtosis.mean',
			'lowlevel.spectral_kurtosis.stdev',
			'lowlevel.spectral_rms.mean',
			'lowlevel.spectral_rms.stdev',
			'lowlevel.spectral_rolloff.mean',
			'lowlevel.spectral_rolloff.stdev',
			'lowlevel.spectral_skewness.mean',
			'lowlevel.spectral_skewness.stdev',
			'lowlevel.spectral_spread.mean',
			'lowlevel.spectral_spread.stdev',
			'lowlevel.spectral_strongpeak.mean',
			'lowlevel.spectral_strongpeak.stdev',
			'lowlevel.zerocrossingrate.mean',
			'lowlevel.zerocrossingrate.stdev',
			'rhythm.beats_count',
			'rhythm.beats_loudness.mean',
			'rhythm.beats_loudness.stdev',
			'rhythm.bpm',
			'rhythm.bpm_histogram_first_peak_bpm',
			'rhythm.bpm_histogram_first_peak_weight',
			'rhythm.bpm_histogram_second_peak_bpm',
			'rhythm.bpm_histogram_second_peak_spread',
			'rhythm.bpm_histogram_second_peak_weight',
			'rhythm.danceability',
			'rhythm.onset_rate',
			'tonal.chords_changes_rate',
			'tonal.chords_number_rate',
			'tonal.chords_strength.mean',
			'tonal.chords_strength.stdev',
			'tonal.hpcp_crest.mean',
			'tonal.hpcp_crest.stdev',
			'tonal.hpcp_entropy.mean',
			'tonal.hpcp_entropy.stdev',
			'tonal.key_edma.strength',
			'tonal.key_krumhansl.strength',
			'tonal.key_temperley.strength',
			'tonal.tuning_diatonic_strength',
			'tonal.tuning_equal_tempered_deviation',
			'tonal.tuning_frequency',
			'tonal.tuning_nontempered_energy_ratio']

FEATURE_SUBSET = ['lowlevel.average_loudness',
			'lowlevel.dissonance.mean',
			'lowlevel.dissonance.stdev',
			'lowlevel.dynamic_complexity',
			'lowlevel.hfc.mean',
			'lowlevel.hfc.stdev',
			'lowlevel.loudness_ebu128.integrated',
			'lowlevel.loudness_ebu128.loudness_range',
			'lowlevel.loudness_ebu128.momentary.mean',
			'lowlevel.loudness_ebu128.momentary.stdev',
			'lowlevel.loudness_ebu128.short_term.mean',
			'lowlevel.loudness_ebu128.short_term.stdev',
			'lowlevel.melbands_crest.mean',
			'lowlevel.melbands_crest.stdev',
			'lowlevel.melbands_flatness_db.mean',
			'lowlevel.melbands_flatness_db.stdev',
			'lowlevel.melbands_kurtosis.mean',
			'lowlevel.melbands_kurtosis.stdev',
			'lowlevel.melbands_skewness.mean',
			'lowlevel.melbands_skewness.stdev',
			'lowlevel.melbands_spread.mean',
			'lowlevel.melbands_spread.stdev',
			'lowlevel.zerocrossingrate.mean',
			'lowlevel.zerocrossingrate.stdev',
			'rhythm.beats_count',
			'rhythm.beats_loudness.mean',
			'rhythm.beats_loudness.stdev',
			'rhythm.bpm',
			'rhythm.bpm_histogram_first_peak_bpm',
			'rhythm.bpm_histogram_first_peak_weight',
			'rhythm.bpm_histogram_second_peak_bpm',
			'rhythm.bpm_histogram_second_peak_spread',
			'rhythm.bpm_histogram_second_peak_weight',
			'rhythm.danceability',
			'rhythm.onset_rate']

FEATURE_SUBSET_2 = ['lowlevel.average_loudness',
			'lowlevel.dissonance.mean',
			'lowlevel.dissonance.stdev',
			'lowlevel.dynamic_complexity',
			'lowlevel.melbands_crest.mean',
			'lowlevel.melbands_crest.stdev',
			'lowlevel.melbands_flatness_db.mean',
			'lowlevel.melbands_flatness_db.stdev',
			'lowlevel.melbands_kurtosis.mean',
			'lowlevel.melbands_kurtosis.stdev',
			'lowlevel.melbands_skewness.mean',
			'lowlevel.melbands_skewness.stdev',
			'lowlevel.melbands_spread.mean',
			'lowlevel.melbands_spread.stdev',
			'lowlevel.zerocrossingrate.mean',
			'lowlevel.zerocrossingrate.stdev',
			'rhythm.beats_count',
			'rhythm.beats_loudness.mean',
			'rhythm.beats_loudness.stdev',
			'rhythm.bpm',
			'rhythm.bpm_histogram_first_peak_bpm',
			'rhythm.bpm_histogram_first_peak_weight',
			'rhythm.bpm_histogram_second_peak_bpm',
			'rhythm.bpm_histogram_second_peak_spread',
			'rhythm.bpm_histogram_second_peak_weight',
			'rhythm.danceability',
			'rhythm.onset_rate']

FEATURE_SUBSET_3 = ['lowlevel.average_loudness',
			'lowlevel.dissonance.mean',
			'lowlevel.dissonance.stdev',
			'lowlevel.dynamic_complexity',
			'lowlevel.hfc.mean',
			'lowlevel.hfc.stdev',
			'lowlevel.loudness_ebu128.integrated',
			'lowlevel.loudness_ebu128.loudness_range',
			'lowlevel.loudness_ebu128.momentary.mean',
			'lowlevel.loudness_ebu128.momentary.stdev',
			'lowlevel.loudness_ebu128.short_term.mean',
			'lowlevel.loudness_ebu128.short_term.stdev',
			'lowlevel.zerocrossingrate.mean',
			'lowlevel.zerocrossingrate.stdev',
			'rhythm.beats_count',
			'rhythm.beats_loudness.mean',
			'rhythm.beats_loudness.stdev',
			'rhythm.bpm',
			'rhythm.bpm_histogram_first_peak_bpm',
			'rhythm.bpm_histogram_first_peak_weight',
			'rhythm.bpm_histogram_second_peak_bpm',
			'rhythm.bpm_histogram_second_peak_spread',
			'rhythm.bpm_histogram_second_peak_weight',
			'rhythm.danceability',
			'rhythm.onset_rate']

NOT_FOUND = []

PRECISION_LIST = []
RECALL_LIST = []
ACCURACY_LIST = []
RR_LIST = []
AP_LIST = []

class UserProfile():
	def __init__(self,id):
		self.id = id
		self.songs = []
		self.recommendations = []

	def add_song(self,track_id,play_count):
		self.songs.append((track_id,play_count))

	def check_songs(self):
		print(self.songs)

def save_to_json(path,json_data):
	print('inside save to json')
	with open(path, 'w') as json_file:
		json.dump(json_data, json_file)

def compute_feature_vector(song_file):
	# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
	features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
												  rhythmStats=['mean', 'stdev'],
												  tonalStats=['mean', 'stdev'])('previews/' + song_file)

	# See all feature names in the pool in a sorted order
	feature_names = sorted(features.descriptorNames())

	selected_features = []
	selected_features.append(features['lowlevel.average_loudness'])
	selected_features.append(features['rhythm.bpm'])
	selected_features.append(features['lowlevel.dissonance.mean'])
	selected_features.append(features['lowlevel.zerocrossingrate.mean'])
	selected_features.append(features['lowlevel.spectral_energy.mean'])
	fv = np.array(selected_features)
	print(fv)

	return fv

def default(obj):
	if type(obj).__module__ == np.__name__:
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return obj.item()
	raise TypeError('Unknown type:', type(obj))

def extract_features(file_path):
	# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
	print('extracting features for {}'.format(file_path))
	features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
												  rhythmStats=['mean', 'stdev'],
												  tonalStats=['mean', 'stdev'])(file_path)

	feature_dict = {}
	feature_names = sorted(features.descriptorNames())
	for name in feature_names:
		feature_dict[name] = features[name]
	json_data = json.dumps(feature_dict, default=default)
	return json_data

def generate_feature_files(origin_folder, destination_folder):
	
	with open(TRACKLIST_FILE, 'r') as file:
		lines = file.readlines()
		sample_files = []
		for line in lines:
			sample_files.append(line.replace('\n',''))

	dir = os.path.abspath(os.getcwd()) + '/' + destination_folder
	already_existing_data = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			print('{} already exists'.format(name.split('.')[0]))
			already_existing_data.append(name.split('.')[0])

	dir = os.path.abspath(os.getcwd()) + '/' + origin_folder
	for root, dirs, files in os.walk(dir):
		for name in files:
			if name.split('.')[0] in sample_files:
				if name.split('.')[0] not in already_existing_data:
					print('inside if')
					file = os.path.join(root, name)
					try:
						json_data = extract_features(file)
						with open(destination_folder + '/' + name.split('.')[0] + '.json', 'w') as json_file:
							json.dump(json_data, json_file)
					except:
						print("couldn't extract features for {}".format(file))
						with open('no_features.txt', 'a') as no_features_file:
							no_features_file.write(file + '\n')
					
				else:
					print('already existing')
			else:
				print('outside if')

def get_feature_file(track_id):
	try:
		path = FEATURE_FILES_FOLDER + '/' + track_id + '.json'
		with open(path) as json_file:
			json_data = json.loads(json.load(json_file))
			return json_data
	except:
		try:
			if track_id not in NOT_FOUND:
				path = FEATURE_FILES_FOLDER + '/' + track_id + '.json'
				audio_path = 'msd/audio/{}/{}/{}/{}.mp3'.format(track_id[2],track_id[3],track_id[4],track_id)
				json_data = extract_features(audio_path)
				with open(path, 'w') as json_file:
					json.dump(json_data, json_file)
				
				with open(path) as json_file:
					json_data = json.loads(json.load(json_file))
				
				return json_data
			else:
				return None

		except Exception as e:
			print("couldn't extract features for {}: {}".format(track_id,e))
			if track_id not in NOT_FOUND:
				NOT_FOUND.append(track_id)
		return None

def get_penultimate(track_id=None):
	penultimate_path = PENULTIMATE_FOLDER + '/' + track_id + '.json'
	try:
		with open(penultimate_path, 'r') as json_file:
			json_data = json.loads(json.load(json_file))
			return json_data
	except:
		try:
			if track_id not in NOT_FOUND:
				audio_path = 'msd/audio/{}/{}/{}/{}.mp3'.format(track_id[2],track_id[3],track_id[4],track_id)
				penultimate = penultimate_features(audio_path, model='MSD_musicnn_big', print_tags=False)
				json_data = json.dumps(penultimate, default=default)
				
				with open(penultimate_path, 'w') as json_file:
					json.dump(json_data, json_file)
				
				with open(penultimate_path) as json_file:
					json_data = json.loads(json.load(json_file))
				
				return json_data

			else:
				return None

		except Exception as e:
			print("couldn't extract penultimate for {}: {}".format(track_id,e))
			if track_id not in NOT_FOUND:
				NOT_FOUND.append(track_id)

		return None

def get_tags(track_id=None):
	tag_path = TAGS_FOLDER + '/' + track_id + '.txt'
	tags = []
	try:
		with open(tag_path, 'r') as file:
			lines = file.readlines()
			for line in lines:
				tags.append(line.replace('\n',''))
	except:
		try:
			if track_id not in NOT_FOUND:
				audio_path = 'msd/audio/{}/{}/{}/{}.mp3'.format(track_id[2],track_id[3],track_id[4],track_id)
				tags = top_tags(audio_path, model='MSD_musicnn_big', topN=20, print_tags=False)
				with open(tag_path, 'w') as file:
					for tag in tags:
						file.write(tag + '\n')
		except:
			print('file {}.mp3 not found'.format(track_id))
			if track_id not in NOT_FOUND:
				NOT_FOUND.append(track_id)

	return tags

def cos_sim(a, b):
	dot_product = np.dot(a,b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	if norm_a > 0 and norm_b > 0:
		return dot_product / (norm_a * norm_b)
	else:
		print('inside else in cos_sim')
		return 1

def euclidean_distance(a, b):
	return np.sqrt(np.sum((a-b) ** 2))

def compute_similarity(a,b,method=None):
	if method == 'euclidean_distance':
		return 1/euclidean_distance(a,b)
	else:
		return cos_sim(a,b)

def tag_similarity(a,b,max_tags=None):
	similarity = 0
	if max_tags:
		a = a[:max_tags]
		b = b[:max_tags]
	for tag in a:
		if tag in b:
			similarity += 1
	return similarity

def evaluate_user_recommendations(u):
	TP = 0
	FN = 0
	P = 0

	first_correct_answer_index = 0
	results = []

	test_songs = []
	with open(TEST_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			test_song = line.replace('\n','')
			if test_song not in NOT_FOUND:
				test_songs.append(test_song)

	with open(TRIPLETS_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			user, song, play_count = line.replace('\n','').split('\t')
			if song in test_songs:
				if int(play_count) > 0:
					if user == u.id:
						P += 1
						results.append(1)
						if song in u.recommendations:
							if TP == 0:
								first_correct_answer_index = P
							TP += 1
						else:
							FN += 1
					else:
						results.append(0)

	FP = len(u.recommendations) - TP
	N = len(test_songs) - P
	TN = N - FP

	if first_correct_answer_index > 0:
		RR = 1/first_correct_answer_index
	else:
		RR = 0

	precisions = []
	for i in range(len(results)):
		pos = i+1
		precisions.append(sum(results[:pos])/pos)
	AP = sum(precisions)/len(precisions)

	print('P: {}'.format(P))
	print('N: {}'.format(N))
	print('TP: {}'.format(TP))
	print('FN: {}'.format(FN))
	print('FP: {}'.format(FP))
	print('TN: {}'.format(TN))

	precision, recall, accuracy = 0, 0, 0

	if (TP+FP) > 0:
		precision = TP/(TP+FP)
		print('Precision: {}'.format(precision))
	if (TP+FN) > 0:
		recall = TP/(TP+FN)
		print('Recall: {}'.format(recall))
	if (P+N) > 0:
		accuracy = (TP+TN)/(P+N)
		print('Accuracy: {}'.format(accuracy))
	if P > 0 and len(u.recommendations) > 0:
		PRECISION_LIST.append(precision)
		RECALL_LIST.append(recall)
		ACCURACY_LIST.append(accuracy)
		RR_LIST.append(RR)
		AP_LIST.append(AP)

	print('Reciprocal Rank: {}'.format(RR))
	print('Average Precision: {}'.format(AP))

	json_data = {
		"id": u.id,
		"recommendations": u.recommendations,
		"metrics": {
			"P": P,
			"N": N,
			"TP": TP,
			"FN": FN,
			"FP": FP,
			"TN": TN,
			"precision": precision,
			"recall": recall,
			"accuracy": accuracy,
			"reciprocal rank": RR,
			"average precision": AP
		}
	}

	save_to_json('evaluation/fbcs/{}.json'.format(u.id),json_data)

def normalize_feature_value(value,feature_range):
	normalized_value = (value - feature_range[0]) / (feature_range[1] - feature_range[0])
	return normalized_value

def generate_user_recommendations(user, features, test_songs, selected_features, min_max_features):
	similarities = {}

	start = datetime.now()

	for song in user.songs:
		for test_song in test_songs:
			if test_song not in NOT_AVAILABLE_SONGS:
				features1 = get_feature_file(song[0])
				features2 = get_feature_file(test_song)
				if features1 and features2:
					selected_features1 = []
					selected_features2 = []
					for feature in FEATURES:
						selected_features1.append(features1[feature])
						selected_features2.append(features2[feature])
					fv1 = np.array(selected_features1)
					fv2 = np.array(selected_features2)
					similarity = compute_similarity(fv1,fv2,'cos_sim')
				else:
					similarity = 0
				if test_song in similarities:
						if similarity > similarities[test_song]:
							similarities[test_song] = similarity
				else:
					similarities[test_song] = similarity

	print('Time after comparing songs: {}'.format(datetime.now()-start))

	similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
	for similarity in similarities[:RECOMMENDATIONS_N]:
		if similarity[1] > 0:
			user.recommendations.append(similarity[0])

	print('Time after getting recommendations: {}'.format(datetime.now()-start))

def generate_user_recommendations_by_penultimate(user, features):
	similarities = {}
	test_songs = []

	start = datetime.now()

	with open(TEST_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			test_songs.append(line.replace('\n',''))

	for test_song in test_songs:
		features2 = get_penultimate(test_song)
		for song in user.songs:
			features1 = get_penultimate(song[0])
			if features1 and features2:
				fv1 = np.empty((0))
				fv2 = np.empty((0))
				for feature in features:
					fv1 = np.concatenate((fv1,features1[feature]))
					fv2 = np.concatenate((fv2,features2[feature]))
				similarity = compute_similarity(fv1,fv2,'cos_sim')
			else:
				similarity = 0
			if test_song in similarities:
				if similarity > similarities[test_song]:
					similarities[test_song] = similarity
			else:
				similarities[test_song] = similarity

	print('Time after comparing songs: {}'.format(datetime.now()-start))

	similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
	for similarity in similarities[:RECOMMENDATIONS_N]:
		if similarity[1] > 0:
			user.recommendations.append(similarity[0])

	print('Time after getting recommendations: {}'.format(datetime.now()-start))

def generate_user_recommendations_by_tag(user):
	similarities = {}
	test_songs = []

	with open(TEST_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			test_songs.append(line.replace('\n',''))

	n_test_songs = 0
	for test_song in test_songs:
		tags2 = get_tags(test_song)
		if tags2:
			n_test_songs += 1
			for song in user.songs:
				tags1 = get_tags(song[0])
				if tags1 and tags2:
					similarity = tag_similarity(tags1,tags2)
				else:
					similarity = 0
				if test_song in similarities:
					if similarity > similarities[test_song]:
						similarities[test_song] = similarity
				else:
					similarities[test_song] = similarity

	similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
	for similarity in similarities[:RECOMMENDATIONS_N]:
		if similarity[1] > 0:
			user.recommendations.append(similarity[0])

def recommender_system(features=None):
	global PRECISION_LIST
	global RECALL_LIST
	global ACCURACY_LIST
	global RR_LIST
	global AP_LIST

	if not features:
		features = FEATURES

	users = {}
	train_songs = []

	with open(TRAIN_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			train_songs.append(line.replace('\n',''))

	with open(TRIPLETS_FILE, 'r') as file:
		lines = file.readlines()
		for line in lines:
			user, song, play_count = line.replace('\n','').split('\t')
			if song in train_songs:
				if int(play_count) > 1:
					if user in users:
						u = users[user]
						u.add_song(song,play_count)
					else:
						u = UserProfile(user)
						u.add_song(song,play_count)
						users[user] = u
				else:
					pass
			else:
				pass

	n_users = 0
	if FEATURE_METHOD:
		print(features)
		test_songs = []

		with open(TEST_FILE, 'r') as file:
			lines = file.readlines()
			for line in lines:
				test_songs.append(line.replace('\n',''))

		selected_features = {}
		min_max_features = {}
		for test_song in test_songs:
			if test_song not in NOT_AVAILABLE_SONGS:
				feature_file = get_feature_file(test_song)
				if feature_file:
					for feature in features:
						if feature in selected_features:
							selected_features[feature].append(feature_file[feature])
						else:
							selected_features[feature] = [feature_file[feature]]

	MAX_USERS = 100
	for user in users:
		if n_users >= MAX_USERS:
			break
		n_users += 1
		print('{}/{} users'.format(n_users,MAX_USERS))

		user = users[user]
		print(user.songs)

		path = 'evaluation/fbcs/{}.json'.format(user.id)
		if os.path.isfile(path):
			print('file {} exists'.format(path))
		else:
			print("file {} doesn't exist".format(path))
			if FEATURE_METHOD:
				generate_user_recommendations(user, features, test_songs, selected_features, min_max_features)
				evaluate_user_recommendations(user)
			if TAG_METHOD:
				generate_user_recommendations_by_tag(user)
				print('user recommendations: {}'.format(user.recommendations))
				evaluate_user_recommendations(user)
			if PENULTIMATE_METHOD:
				features = ['max']
				generate_user_recommendations_by_penultimate(user,features)
				print('user recommendations: {}'.format(user.recommendations))
				evaluate_user_recommendations(user)

	average_precision = sum(PRECISION_LIST)/len(PRECISION_LIST)
	average_recall = sum(RECALL_LIST)/len(RECALL_LIST)
	average_accuracy = sum(ACCURACY_LIST)/len(ACCURACY_LIST)
	MRR = sum(RR_LIST)/len(RR_LIST)
	MAP = sum(AP_LIST)/len(AP_LIST)

	print('')
	print('Users: {}/{}'.format(len(PRECISION_LIST),min(MAX_USERS,len(users))))
	print('Average precision: {}'.format(average_precision))
	print('Average recall: {}'.format(average_recall))
	print('Average accuracy: {}'.format(average_accuracy))
	print('MRR: {}'.format(MRR))
	print('MAP: {}'.format(MAP))
	print('\n')

	PRECISION_LIST = []
	RECALL_LIST = []
	ACCURACY_LIST = []
	RR_LIST = []
	AP_LIST = []

#recommender_system(FEATURE_SUBSET_2)
#recommender_system(FEATURE_SUBSET_3)
#recommender_system(['lowlevel.average_loudness','rhythm.bpm','lowlevel.dissonance.mean','lowlevel.zerocrossingrate.mean','lowlevel.spectral_energy.mean'])
#recommender_system(['lowlevel.dissonance.mean','lowlevel.zerocrossingrate.mean','lowlevel.spectral_energy.mean'])
#recommender_system(['lowlevel.average_loudness'])
#recommender_system(['rhythm.bpm'])
#recommender_system(['lowlevel.dissonance.mean'])
#recommender_system(['lowlevel.zerocrossingrate.mean'])
#recommender_system(['lowlevel.spectral_energy.mean'])
recommender_system(FEATURE_SUBSET)

print(NOT_FOUND)
print(len(NOT_FOUND))

#generate_feature_files(PREVIEWS_FOLDER, FEATURE_FILES_FOLDER)

#tags = get_tags()
#print(tags)
