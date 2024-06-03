# Adapted implementation of Malsburg
# for details see R implementation: https://github.com/tmalsburg/scanpath/blob/master/scanpath/R/scanpath.R
from utils import *


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
            central_subject_id = subject_ids[central_sp_index]
            min_scasim_score = min(dissimilarity_matrix, key=lambda x: x[1])[1]
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

        for sn_id in tqdm(sn_list): # 5456
            # Filter eyemovement data for the given sentence
            eyemovement_df_sn = eyemovement_df[eyemovement_df['sentenceid'] == sn_id]

            # notice: Each sentence is recorded multiple times in file |word_info_df|.
            sn = word_info_df[word_info_df.sentenceid == sn_id]
            sn = sn[sn['list'] == sn.list.values.tolist()[0]]
            # compute word length for each word
            sn_word_len = compute_word_length_celer(sn.WORD_LEN.values)  # a numpy array for the current sentence

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

            # if a sentence was read by more than 1 subject
            if len(grouped_scanpaths) > 1:
                dissimilarity_matrix = []  # Compute dissimilarity matrix
                for i, sp1 in enumerate(grouped_scanpaths):
                    total_dissimilarity = 0
                    count = 0
                    # some sentences in CELER are read by > 300 subjects
                    # use only 20 subjects for the sp comparison
                    for j in range(i + 1, min(i + 21, len(grouped_scanpaths))):
                        if i != j:  # Exclude self-comparison
                            dissimilarity = scasim(sp1, grouped_scanpaths[j])  # Calculate dissimilarity
                            total_dissimilarity += dissimilarity
                            count += 1
                    if count > 0:
                        average_dissimilarity = total_dissimilarity / count
                        dissimilarity_matrix.append((i, average_dissimilarity))

                central_sp_index = min(dissimilarity_matrix, key=lambda x: x[1])[0]
                central_scanpath = grouped_scanpaths[central_sp_index]
                #  central_subject_id = subject_ids[central_sp_index]
                # min_scasim_score = min(dissimilarity_matrix, key=lambda x: x[1])[1]
                central_sp_per_sent[sn_id] = central_scanpath

            # if the sentence was read by only 1 subject - no need to calculate scasim, take this sp directly
            elif len(grouped_scanpaths) == 1:
                central_sp_per_sent[sn_id] = grouped_scanpaths[0]
            else:  # in case the sent was removed after pre-processing
                continue
    return central_sp_per_sent


def compute_scasim(central_sp_path, scanpath):
    with open(central_sp_path, "r") as f:
        data_str = f.read()
        central_scanpaths = ast.literal_eval(data_str)

    for key, value in central_scanpaths.items():
        central_scanpaths[key] = [(int(x[0]), x[1], int(x[-1])) for x in value]

    # For CELER map the sentence names to their corresponding sent ids
    if "CELER" in central_sp_path:
        with open('././data_splits/CELER_sent_dict.txt', "r") as f:
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)

        new_central_sp = {}
        for key, value in central_scanpaths.items():
            # Check if the key exists in the mapping dictionary
            if key in sent_dict:
                # Get the corresponding ID from the mapping
                new_key = sent_dict[key]
                new_central_sp[new_key] = value
        central_scanpaths = new_central_sp

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
                current_sp = [(location[i], 0, duration[i]) for i in range(len(location))]
                formatted_sp.append(current_sp)
                # Compute similarity between the current sp and the most central sp
                score = scasim(current_sp, central_scanpath)
                similarity_scores.append(int(score))
    return similarity_scores
