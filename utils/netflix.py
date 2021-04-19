import os
import codecs
from tqdm import tqdm
from utils.util import *

datadir = '/home/xx/data/netflix'
pro_dir = os.path.join(datadir, 'large')
if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)


np.random.seed(2020)
n_heldout_users = 40000
######################################################
lst_interactions = []
for curr_split in [1, 2, 3, 4]:

	curr_split_path = os.path.join(datadir, "combined_data_{}.txt".format( curr_split ))
	with codecs.open(curr_split_path, "r", encoding = "utf-8", errors = "ignore") as inFile:
		currMovieID = None
		doc = inFile.readlines()
		for line in tqdm(doc, desc = "File {:d}/4".format(curr_split)):
			if (len(line) > 1):
				line_split = line.split(",")
				# If line has 3 components, it is a 'user_id,rating,date' row
				if (len(line_split) == 3 and currMovieID != None):
					userID = line_split[0]
					rating = float(line_split[1])
					if (rating >= 4):
						lst_interactions.append([userID, currMovieID, rating])
				# If line has 1 component, it MIGHT be a 'item_id:' row
				elif (len(line_split) == 1):
					line_split = line.split(":")
					# Confirm it is a 'item_id:' row
					if (len(line_split) == 2):
						currMovieID = line_split[0]
					else:
						print("Unexpected Row: '{}'".format( line ))
				else:
					print("Unexpected Row: '{}'".format( line ))
	print("Processed '{}'..".format( curr_split_path ))

print("\n# of interactions: {:,d}".format(len(lst_interactions)))
raw_data = pd.DataFrame(lst_interactions, columns = ["userId", "movieId", "rating"])
######################################################

# raw_data = pd.read_csv(os.path.join(datadir, 'ratings.csv'), header=0)
# raw_data = raw_data[raw_data.rating > 3.5]

## only keep users who have watched at least 5 movies
raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5)
num_users = user_activity.shape[0]
num_items = item_popularity.shape[0]
sparsity = 1. * raw_data.shape[0] / (num_users * num_items)
print("watching events: {0:}, users: {1:}, items: {2:}, sparsity: {3:.3f}%".format(raw_data.shape[0],
                                                                                   num_users,
                                                                                   num_items,
                                                                                   sparsity * 100))

unique_uid = user_activity.index
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]



n_users = unique_uid.size
tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2):(n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data.userId.isin(tr_users)]
## The movies watched by the training users compose of the complete (item) set.
unique_sid = pd.unique(train_plays['movieId'])
np.savetxt(os.path.join(pro_dir, "unique_sid.txt"), unique_sid, fmt = "%s")
print("Number of users:{}, items: {}".format(n_users, len(unique_sid)))

vad_plays = raw_data.loc[raw_data.userId.isin(vd_users)]
## the movies watched by users in validation must be a subset of the complete set.
vad_plays = vad_plays.loc[vad_plays.movieId.isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data.userId.isin(te_users)]
## the movies watched by users in testing dataset must be a subset of the complete set.
test_plays = test_plays.loc[test_plays.movieId.isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

## re-numbering the users (items) from 1 to n (m).
movie2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

def numerize(df, movie2id, user2id):
    uid = list(map(lambda x: user2id[x], df['userId']))
    sid = list(map(lambda x: movie2id[x], df['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

## saving training, validation and test data.
train_data = numerize(train_plays, movie2id, user2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

# Merge validation fold-in set and testing fold-in set with the training data
train_plays_merged = pd.concat([train_plays, vad_plays_tr, test_plays_tr])
train_data_merged = numerize(train_plays_merged, movie2id, user2id)
train_data_merged.to_csv(os.path.join(pro_dir, "train_merged.csv"), index = False)

vad_data_tr = numerize(vad_plays_tr, movie2id, user2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
vad_data_te = numerize(vad_plays_te, movie2id, user2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr, movie2id, user2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
test_data_te = numerize(test_plays_te, movie2id, user2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
