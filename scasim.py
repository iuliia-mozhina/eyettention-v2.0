# Adapted implementation of Malsburg
# for details see R implementation: https://github.com/tmalsburg/scanpath/blob/master/scanpath/R/scanpath.R
from math import acos, sin, cos, pi
from collections import Counter
from utils import *
import ast
import json


def modify_landing_pos(sp_human):
    """Converts elements in each list of sp_human['landing_pos'] (except first & last) to floats with 7 decimals."""
    for i in range(len(sp_human["landing_pos"])):
        landing_pos = sp_human["landing_pos"][i]
        # Convert elements except first and last to floats with 7 decimals
        landing_pos[1:-1] = [float("{:.8f}".format(x)) for x in landing_pos[1:-1]]
        sp_human["landing_pos"][i] = landing_pos

    return sp_human


def convert_sp_to_lists(sp):
    """Converts values in sp_dnn dictionary to regular lists.

  Args:
      sp_dnn (dict): Dictionary with keys "locations" and "durations".
          - Values can be lists of NumPy arrays or other nested structures.

  Returns:
      dict: The modified sp_dnn dictionary with all values converted to lists.
  """

    # Iterate through each key-value pair
    for key, value in sp.items():
        # Check if the value is a list
        if isinstance(value, list):
            # If it's a list, iterate through its elements
            new_value = []
            for element in value:
                # Convert each element to a list (handles NumPy arrays or nested structures)
                new_value.append(element.tolist() if hasattr(element, 'tolist') else element)
            # Update the dictionary with the converted list
            sp[key] = new_value
        else:
            # If the value is not a list, leave it unchanged (handles non-list values)
            pass

    return sp


def sample_random_sp(dataset, sp_human, sent_dict_path=None):
    random_sp = {}
    random_sp["locations"] = []
    random_sp["durations"] = []
    random_sp["landing_pos"] = []
    random_sp["sent_id"] = []

    if dataset == 'BSC':
        dataset_path = '././Data/beijing-sentence-corpus/' 
        bsc_emd_path = os.path.join(dataset_path, 'BSC.EMD/BSC.EMD.txt')
        eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
        info_path = os.path.join(dataset_path, 'BSC.Word.Info.v2.xlsx')
        word_info_df = pd.read_excel(info_path, 'word')

        for sn_id in sp_human["sent_id"]:
            eyemovement_df_sn = eyemovement_df[eyemovement_df['sn'] == sn_id]
            sn = word_info_df[word_info_df.SN == sn_id]
            # sample a random subject
            random_subj_id = eyemovement_df_sn['id'].sample(n=1, random_state=1).iloc[0]
            eyemovement_df_sn = eyemovement_df_sn[eyemovement_df_sn['id'] == random_subj_id]

            for subject_id, subject_data in eyemovement_df_sn.groupby(
                    'id'):
                # Filter out last fixation if it's an outlier (wn=1 and fl=0)
                if subject_data.iloc[-1].wn == 1 and subject_data.iloc[-1].fl == 0:
                    subject_data = subject_data.iloc[:-1]

                sp_fix_loc = [row['fl'] for _, row in subject_data.iterrows()]

                sp_landing_pos = np.modf(sp_fix_loc)[0]
                sp_landing_pos = [-0.0] + sp_landing_pos.tolist() + [-0.0]
                # Convert word-based ordinal positions to token(character)-based ordinal positions
                sp_fix_loc = np.where(np.array(sp_fix_loc) < 0, 0, np.array(sp_fix_loc))
                sp_word_pos = [row['wn'] for _, row in subject_data.iterrows()]

                sp_fix_loc = [np.sum(sn[sn.NW < value].LEN) + np.ceil(sp_fix_loc[count] + 1e-10) for count, value in
                              enumerate(sp_word_pos)]

                sp_fix_loc = [0] + sp_fix_loc + [26]
                sp_fix_loc = [int(round(x)) for x in sp_fix_loc]

                sp_fix_dur = [row['dur'] for _, row in subject_data.iterrows()]
                sp_fix_dur = [-0.0] + sp_fix_dur + [-0.0]

                # Append data for each subject to the corresponding sentence list
                random_sp["locations"].append(sp_fix_loc)
                random_sp["durations"].append(sp_fix_dur)
                random_sp["landing_pos"].append(sp_landing_pos)
                random_sp["sent_id"].append(sn_id)

    elif dataset == "CELER":
        eyemovement_df = pd.read_csv('././Data/celer/sent_fix.tsv', delimiter='\t')  
        eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace(
            '\t(.*)', '', regex=True)
        word_info_df = pd.read_csv('drive/MyDrive/celer/sent_ia.tsv', delimiter='\t')
        word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

        with open(sent_dict_path, "r") as f:  # '././data_splits/CELER_sent_dict.txt'
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)

        for sn_id in sp_human["sent_id"]:
            print(sn_id)
            for key, value in sent_dict.items():
                if value == sn_id:
                    sentence_name = key
                    break
            print(sentence_name)
            eyemovement_df_sn = eyemovement_df[eyemovement_df['sentenceid'] == sentence_name]
            sn = word_info_df[word_info_df.sentenceid == sentence_name]
            sn = sn[sn['list'] == sn.list.values.tolist()[0]]

            sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
            # nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
            if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
                # extra inverted commas at the end of the sentence
                sn_str = sn_str[:-3] + sn_str[-1:]
            if sn_id == '1987/w7_085/w7_085.200-18':
                sn_str = sn_str[:43] + sn_str[44:]
            sn_len = len(sn_str.split())

            # sample a random subject
            random_subj_id = eyemovement_df_sn['list'].sample(1).iloc[0]
            eyemovement_df_sn = eyemovement_df_sn[eyemovement_df_sn['list'] == random_subj_id]

            for subject_id, subject_data in eyemovement_df_sn.groupby(
                    'list'):
                sub_df = subject_data.copy()
                # remove fixations on non-words
                sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']

                # prepare decoder input and output
                sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

                # dataset is noisy -> sanity check
                # 1) check if recorded fixation duration are within reasonable limits
                # Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
                outlier_indx = np.where(sp_fix_dur < 50)[0]
                if outlier_indx.size > 0:
                    for out_idx in range(len(outlier_indx)):
                        outlier_i = outlier_indx[out_idx]
                        merge_flag = False

                        # outliers are commonly found in the fixation of the last record and the first record, and are removed directly
                        if outlier_i == len(sp_fix_dur) - 1 or outlier_i == 0:
                            merge_flag = True

                        else:
                            if outlier_i - 1 >= 0 and merge_flag == False:
                                # try to merge with the left fixation
                                if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                    outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                    sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                    merge_flag = True

                            if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                                # try to merge with the right fixation
                                if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                    outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                    sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                    merge_flag = True

                        sp_word_pos = np.delete(sp_word_pos, outlier_i)
                        sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                        sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                        sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                        outlier_indx = outlier_indx - 1

                # 2) scanpath too long, remove outliers, speed up the inference
                if len(sp_word_pos) > 50:  # 72/10684
                    continue
                # 3)scanpath too short for a normal length sentence
                if len(sp_word_pos) <= 1 and sn_len > 10:
                    continue

                # 4) check landing position feature
                # assign missing value to 'nan'
                sp_fix_loc = np.where(sp_fix_loc == '.', np.nan, sp_fix_loc)
                # convert string of number of float type
                sp_fix_loc = [float(i) for i in sp_fix_loc]

                # Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
                if np.nanmax(sp_fix_loc) > 35:
                    missing_idx = np.where(np.array(sp_fix_loc) > 5)[0]
                    for miss in missing_idx:
                        if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                            sp_fix_loc[miss] = np.nan
                        else:
                            print('Landing position calculation error. Unknown cause, needs to be checked')

                # Append data for each subject to the corresponding sentence list
                sp_ordinal_pos = sp_word_pos.astype(int)

                sp_ordinal_pos = [0] + sp_ordinal_pos.tolist() + [23]
                sp_fix_dur = [-0.0] + sp_fix_dur.tolist() + [-0.0]
                sp_fix_loc = [-0.0] + sp_fix_loc + [-0.0]

                random_sp["locations"].append(sp_ordinal_pos)
                random_sp["durations"].append(sp_fix_dur)
                random_sp["landing_pos"].append(sp_fix_loc)
                random_sp["sent_id"].append(sn_id)

    return random_sp


def filter_sp(sp):
    # evaluate only on sentences that were read by multiple users
    sentence_counts = Counter(sp["sent_id"])
    filtered_sent_ids = []
    filtered_durations = []
    filtered_locations = []
    filtered_landing_pos = []
    for sent_id, location, duration, land_pos in zip(sp["sent_id"], sp["locations"], sp["durations"],
                                                     sp["landing_pos"]):
        if sentence_counts.get(sent_id, 0) > 1:
            filtered_sent_ids.append(sent_id)
            filtered_locations.append(location)
            filtered_durations.append(duration)
            filtered_landing_pos.append(land_pos)

    sp["sent_id"] = filtered_sent_ids
    sp["durations"] = filtered_durations
    sp["locations"] = filtered_locations
    sp["landing_pos"] = filtered_landing_pos
    return sp
    

def which_min(*l):
    mi = 0
    for i, e in enumerate(l):
        if e < l[mi]:
            mi = i
    return mi


def scasim(s, t, modulator=0.83):
    # modulator 0.83 (by default) - specifies how spatial distances between fixations
    # #' are assessed.  When set to 0, any spatial divergence of two
    # #' compared scanpaths is penalized independently of its degree.  When
    # #' set to 1, the scanpaths are compared only with respect to their
    # #' temporal patterns.  The default value approximates the sensitivity
    # #' to spatial distance found in the human visual system.

    # Prepare matrix:
    m, n = len(s), len(t)
    d = [list(map(lambda i: 0, range(n + 1))) for _ in range(m + 1)]

    # Sequence alignment
    acc = 0
    # loop over fixations in scanpath s:
    for i in range(1, m + 1):
        acc += s[i - 1][2]
        d[i][0] = acc

    # loop over fixations in scanpath t:
    acc = 0
    for j in range(1, n + 1):
        acc += t[j - 1][2]
        d[0][j] = acc

    # Compute similarity:
    for i in range(n):
        for j in range(m):
            # calculating angle between fixation targets:
            slon = s[j][0] / (180 / pi)  # longitude (x-axis)
            tlon = t[i][0] / (180 / pi)
            slat = s[j][1] / (180 / pi)  # latitude (y-axis)
            tlat = t[i][1] / (180 / pi)

            angle = acos(sin(slat) * sin(tlat) + cos(slat) * cos(tlat) * cos(slon - tlon)) * (180 / pi)

            # approximation of cortical magnification:
            mixer = modulator ** angle

            # cost for substitution:
            cost = (abs(t[i][2] - s[j][2]) * mixer +
                    (t[i][2] + s[j][2]) * (1.0 - mixer))

            # select optimal edit operation
            ops = (d[j][i + 1] + s[j][2],
                   d[j + 1][i] + t[i][2],
                   d[j][i] + cost)

            mi = which_min(*ops)

            d[j + 1][i + 1] = ops[mi]

    return d[-1][-1]


def compute_central_sp(dataset, dataset_path):
    """
    Calculates the most central scanpath for all sentences in the specified dataset.
    The most central scanpath is the sp with the smallest average dissimilarity from every other sp for a given sentence.
    """
    central_sp_per_sent = {}

    if dataset == 'BSC':
        bsc_emd_path = os.path.join(dataset_path, 'BSC.EMD/BSC.EMD.txt')
        eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
        info_path = os.path.join(dataset_path, 'BSC.Word.Info.v2.xlsx')
        word_info_df = pd.read_excel(info_path, 'word')
        sn_list = eyemovement_df['sn'].unique().tolist()

        for sn_id in sn_list:
            # Filter eyemovement data for the given sentence
            eyemovement_df_sn = eyemovement_df[eyemovement_df['sn'] == sn_id]
            sn = word_info_df[word_info_df.SN == sn_id]

            grouped_scanpaths = []
            subject_ids = []
            for subject_id, subject_data in eyemovement_df_sn.groupby(
                    'id'):  # Group by subject ID and prepare scanpaths in the correct format (x-coordinate, y-coordinate, duration)
                # Filter out last fixation if it's an outlier (wn=1 and fl=0)
                if subject_data.iloc[-1].wn == 1 and subject_data.iloc[-1].fl == 0:
                    subject_data = subject_data.iloc[:-1]

                # Convert fixation data into (x, y, dur) tuples
                sp_fix_loc = [row['fl'] for _, row in subject_data.iterrows()]
                # Convert word-based ordinal positions to token(character)-based ordinal positions
                sp_fix_loc = np.where(np.array(sp_fix_loc) < 0, 0, np.array(sp_fix_loc))
                sp_word_pos = [row['wn'] for _, row in subject_data.iterrows()]

                sp_fix_loc = [np.sum(sn[sn.NW < value].LEN) + np.ceil(sp_fix_loc[count] + 1e-10) for count, value in
                              enumerate(sp_word_pos)]
                subject_data['sp_fix_loc'] = sp_fix_loc

                # Create scanpath
                scanpath = [(0, 0, -0.0)]  # Add initial tuple
                scanpath += [(row['sp_fix_loc'], 0, row['dur']) for _, row in subject_data.iterrows()]
                scanpath.append((26, 0, -0.0))  # Add final tuple

                grouped_scanpaths.append(scanpath)
                subject_ids.append(subject_id)

            dissimilarity_matrix = []
            for i, sp1 in enumerate(grouped_scanpaths):
                total_dissimilarity = 0
                count = 0
                for j, sp2 in enumerate(grouped_scanpaths):
                    if i != j:  # Exclude self-comparison
                        dissimilarity = scasim(sp1, sp2)  # Calculate dissimilarity
                        total_dissimilarity += dissimilarity
                        count += 1
                if count > 0:
                    average_dissimilarity = total_dissimilarity / count
                    dissimilarity_matrix.append((i, average_dissimilarity))

            central_sp_index = min(dissimilarity_matrix, key=lambda x: x[1])[0]
            central_scanpath = grouped_scanpaths[central_sp_index]
            central_sp_per_sent[sn_id] = central_scanpath

    elif dataset == "CELER":
        eye_df_path = os.path.join(dataset_path, 'data_v2.0/sent_fix.tsv')
        eyemovement_df = pd.read_csv(eye_df_path, delimiter='\t')
        eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace(
            '\t(.*)', '', regex=True)
        word_info_path = os.path.join(dataset_path, 'data_v2.0/sent_ia.tsv')
        word_info_df = pd.read_csv(word_info_path, delimiter='\t')
        word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

        reader_list = celer_load_native_speaker()
        sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

        for sn_id in tqdm(sn_list):  # 5456
            # Filter eyemovement data for the given sentence
            eyemovement_df_sn = eyemovement_df[eyemovement_df['sentenceid'] == sn_id]

            # notice: Each sentence is recorded multiple times in file |word_info_df|.
            sn = word_info_df[word_info_df.sentenceid == sn_id]
            sn = sn[sn['list'] == sn.list.values.tolist()[0]]

            sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
            # nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
            if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
                # extra inverted commas at the end of the sentence
                sn_str = sn_str[:-3] + sn_str[-1:]
            if sn_id == '1987/w7_085/w7_085.200-18':
                sn_str = sn_str[:43] + sn_str[44:]
            sn_len = len(sn_str.split())

            grouped_scanpaths = []
            subject_ids = []
            for subject_id, subject_data in eyemovement_df_sn.groupby(
                    'list'):
                # filter out the L2 subjects when computing the most central sp
                if subject_id in reader_list:
                    # Preprocess data
                    sub_df = subject_data.copy()

                    # remove fixations on non-words
                    sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
                    sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

                    # 1) Fixation duration outlier handling
                    outlier_indx = np.where(sp_fix_dur < 50)[0]
                    if outlier_indx.size > 0:
                        for out_idx in range(len(outlier_indx)):
                            outlier_i = outlier_indx[out_idx]
                            merge_flag = False

                            # Handle edge cases (first/last fixations)
                            if outlier_i == len(sp_fix_dur) - 1 or outlier_i == 0:
                                merge_flag = True

                            else:
                                # Attempt to merge with neighboring fixations
                                if outlier_i - 1 >= 0 and merge_flag == False:
                                    if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                        outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                        sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                        merge_flag = True

                                if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                                    if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                        outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                        sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                        merge_flag = True

                            # Remove outliers from all np.arrays and DataFrame
                            sp_word_pos = np.delete(sp_word_pos, outlier_i)
                            sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                            sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                            sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                            outlier_indx = outlier_indx - 1

                    # 2) scanpath too long, remove outliers, speed up the inference
                    if len(sp_word_pos) > 50:  # 72/10684
                        continue
                    # 3)scanpath too short for a normal length sentence
                    if len(sp_word_pos) <= 1 and sn_len > 10:
                        continue

                    sp_word_pos = sp_word_pos.astype(float)
                    sp_fix_dur = sp_fix_dur.astype(float)

                    # Create scanpath
                    scanpath = [(0, 0, -0.0)]  # Add initial tuple
                    scanpath += [(pos, 0, dur) for pos, dur in zip(sp_word_pos, sp_fix_dur)]
                    scanpath.append((24, 0, -0.0))  # Add final tuple
                    grouped_scanpaths.append(scanpath)
                    subject_ids.append(subject_id)

                else:
                    continue

            # if a sentence was read by more than 1 subject
            if len(grouped_scanpaths) > 1:
                print(len(grouped_scanpaths))
                dissimilarity_matrix = []  # Compute dissimilarity matrix
                for i, sp1 in enumerate(grouped_scanpaths):
                    total_dissimilarity = 0
                    count = 0
                    for j, sp2 in enumerate(grouped_scanpaths):
                        if i != j:  # Exclude self-comparison
                            dissimilarity = scasim(sp1, sp2)  # Calculate dissimilarity
                            total_dissimilarity += dissimilarity
                            count += 1
                    if count > 0:
                        average_dissimilarity = total_dissimilarity / count
                        dissimilarity_matrix.append((i, average_dissimilarity))

                central_sp_index = min(dissimilarity_matrix, key=lambda x: x[1])[0]
                central_scanpath = grouped_scanpaths[central_sp_index]
                central_sp_per_sent[sn_id] = central_scanpath

            # if the sentence was read by only 1 subject - we exclude this sentence from eval
            else:
                continue

    else:  # Zuco dataset
        zuco_path = './Data/zuco/task2/Matlab_files'  # train / eval only on NR task
        word_info_path = zuco_path + '/Word_Infor.csv'
        word_info_df = pd.read_csv(word_info_path, sep='\t')
        scanpath_path = zuco_path + '/scanpath.csv'
        eyemovement_df = pd.read_csv(scanpath_path, sep='\t')

        sn_list = np.unique(eyemovement_df.sn.values).tolist()  # 300
        for sn_id in tqdm(sn_list):
            # process sentence sequence
            eyemovement_df_sn = eyemovement_df[eyemovement_df.sn == sn_id]
            sn = word_info_df[word_info_df.SN == sn_id]
            sn_str = ' '.join(sn.WORD.values)
            sn_str = '[CLS] ' + sn_str + ' [SEP]'
            sn_len = len(sn_str.split())

            grouped_scanpaths = []
            subject_ids = []
            for subject_id, subject_data in eyemovement_df_sn.groupby(
                    'id'):
                sub_df = subject_data.copy()
                # remove fixations on non-words
                sub_df = sub_df.loc[
                    sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
                # prepare decoder input and output
                sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values

                # check if recorded fixation duration are within reasonable limits
                # Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
                outlier_indx = np.where(sp_fix_dur < 50)[0]
                if outlier_indx.size > 0:
                    for out_idx in range(len(outlier_indx)):
                        outlier_i = outlier_indx[out_idx]
                        merge_flag = False
                        if outlier_i - 1 >= 0 and not merge_flag:
                            # try to merge with the left fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and not merge_flag:
                            # try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[
                                outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        sp_word_pos = np.delete(sp_word_pos, outlier_i)
                        sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                        sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                        sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                        outlier_indx = outlier_indx - 1

                # sanity check
                # scanpath too short for a normal length sentence
                if len(sp_word_pos) <= 1 and sn_len > 10:
                    continue

                sp_word_pos = sp_word_pos.astype(float)
                sp_fix_dur = sp_fix_dur.astype(float)

                # Create scanpath
                scanpath = [(0, 0, -0.0)]  # Add initial tuple
                scanpath += [(pos, 0, dur) for pos, dur in zip(sp_word_pos, sp_fix_dur)]
                scanpath.append((24, 0, -0.0))  # Add final tuple
                grouped_scanpaths.append(scanpath)
                subject_ids.append(subject_id)

            dissimilarity_matrix = []  # Compute dissimilarity matrix
            for i, sp1 in enumerate(grouped_scanpaths):
                total_dissimilarity = 0
                count = 0
                for j, sp2 in enumerate(grouped_scanpaths):
                    if i != j:  # Exclude self-comparison
                        dissimilarity = scasim(sp1, sp2)  # Calculate dissimilarity
                        total_dissimilarity += dissimilarity
                        count += 1
                if count > 0:
                    average_dissimilarity = total_dissimilarity / count
                    dissimilarity_matrix.append((i, average_dissimilarity))

            central_sp_index = min(dissimilarity_matrix, key=lambda x: x[1])[0]
            central_scanpath = grouped_scanpaths[central_sp_index]
            central_sp_per_sent[sn_id] = central_scanpath

    return central_sp_per_sent


def compute_scasim(sp1, sp2):
    similarity_scores = []
    formatted_sp1 = []
    formatted_sp2 = []
    for loc, dur in zip(sp1["locations"], sp1["durations"]):
        try:
            current_sp = [(loc[i], 0, dur[i]) for i in range(len(loc))]
        except IndexError:
            current_sp = [(1, 0, 1)]
        formatted_sp1.append(current_sp)

    for loc, dur in zip(sp2["locations"], sp2["durations"]):
        current_sp = [(loc[i], 0, dur[i]) for i in range(len(loc))]
        formatted_sp2.append(current_sp)

    for i in range(len(formatted_sp1)):
        score = scasim(formatted_sp1[i], formatted_sp2[i])
        similarity_scores.append(int(score))

    return similarity_scores


def compute_central_scasim(central_sp_path, scanpath, sent_dict_path=None):
    # For CELER map the sentence names to their corresponding sent ids
    if "CELER" in central_sp_path:
        with open(sent_dict_path, "r") as f:  # '././data_splits/CELER_sent_dict.txt'
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)

    with open(central_sp_path, "r") as f:
        data_str = f.read()
        central_scanpaths = ast.literal_eval(data_str)

        if "CELER" in central_sp_path:
            new_central_sp = {}
            for key, value in central_scanpaths.items():
                # Check if the key exists in the mapping dictionary
                if key in sent_dict:
                    # Get the corresponding ID from the mapping
                    new_key = sent_dict[key]
                    new_central_sp[new_key] = value
            central_scanpaths = new_central_sp

        central_scanpaths = {int(key): value for key, value in central_scanpaths.items()}

    for key, value in central_scanpaths.items():
        central_scanpaths[key] = [(int(x[0]), x[1], int(x[-1])) for x in value]

    sent_to_indx_dict = {}
    for index, sent_id in enumerate(scanpath['sent_id']):
        if sent_id not in sent_to_indx_dict:
            sent_to_indx_dict[sent_id] = []
        sent_to_indx_dict[sent_id].append(index)

    similarity_scores = []
    formatted_sp = []  # list of lists of tuples
    for sent_id in set(scanpath['sent_id']):  # iterate through sentences in the scanpath dict
        if sent_id in central_scanpaths:
            # Retrieve the corresponding most central scanpath for the current sent_id
            central_scanpath = central_scanpaths[sent_id]
            # Get the locations and durations for the current sentence
            idx = sent_to_indx_dict[sent_id]
            for id in idx:
                location = scanpath['locations'][id]
                duration = scanpath['durations'][id]
                try:
                    current_sp = [(location[i], 0, duration[i]) for i in range(len(location))]
                except IndexError:
                    current_sp = [(1, 0, 1)]
                formatted_sp.append(current_sp)
                # Compute similarity between the current sp and the most central sp
                score = scasim(current_sp, central_scanpath)
                similarity_scores.append(int(score))
    return similarity_scores
