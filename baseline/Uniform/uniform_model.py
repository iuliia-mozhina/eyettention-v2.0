from scasim import *
from utils import *


def uniform_fixation_model(dataset, df, min_dur, max_dur):
    sentence_df = pd.DataFrame(columns=['SN', 'loc', 'dur', 'land_pos'])
    if dataset == "BSC":
        grouped_df = df.groupby('SN')['WORD'].apply(list)
    else:  # CELER
        grouped_df = df.groupby('sentenceid')['WORD_NORM'].apply(list)
    for sentence_num, sentence_words in grouped_df.items():
        num_words = len(sentence_words)
        locations = []
        current_location = 0
        # Generate random saccades
        while current_location != num_words + 1:  # iterate until we reach the EOS token
            saccade = np.random.randint(-num_words, num_words + 2)  # predict a saccade
            new_location = current_location + saccade  # update the fixation location

            # only update the locations if it is valid
            if 0 <= new_location <= num_words + 1:
                locations.append(new_location)
                current_location = new_location
        locations = np.array(locations)
        # Generate random durations
        durations = np.random.randint(min_dur, max_dur + 1, size=len(locations))
        # Generate random landing positions
        land_pos = np.random.uniform(low=0.0, high=1.0, size=len(locations))
        sentence_data = pd.DataFrame(
            {'SN': sentence_num, 'loc': locations, 'dur': durations, 'land_pos': land_pos})
        sentence_df = sentence_df.append(sentence_data)

    sentence_df = sentence_df.reset_index(drop=True)
    return sentence_df


def compute_uniform_bsc():
    bsc_path = '../../Data/beijing-sentence-corpus/'
    info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
    bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')
    eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
    eyemovement_df["land_pos"] = np.modf(eyemovement_df["fl"])[0]

    # For duration prediction determine the min and max values
    # Calculate IQR for the 'dur' column
    Q1 = eyemovement_df["dur"].quantile(0.25)
    Q3 = eyemovement_df["dur"].quantile(0.75)
    IQR = Q3 - Q1

    # Determine bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter durations within the bounds
    filtered_durations = eyemovement_df["dur"][
        (eyemovement_df["dur"] >= lower_bound) & (eyemovement_df["dur"] <= upper_bound)]

    # Determine min_dur and max_dur
    min_dur = filtered_durations.min()
    max_dur = filtered_durations.max()

    df = pd.read_excel(info_path, 'word')
    all_subjects_df = pd.DataFrame()
    for subject_id in range(1, 61):  # 60 subjects in the BSC dataset
        BSC_pred = uniform_fixation_model("BSC", df.copy(), min_dur, max_dur)
        BSC_pred['subj_id'] = subject_id
        all_subjects_df = all_subjects_df.append(BSC_pred)
    all_subjects_df = all_subjects_df.reset_index(drop=True)
    all_subjects_df.to_csv('BSC_uniform_results.csv', index=False)
    return all_subjects_df


def compute_uniform_celer():
    eyemovement_df = pd.read_csv('../../Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace(
        '\t(.*)', '', regex=True)
    word_info_df = pd.read_csv('../../Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # native speakers only
    sub_metadata_path = '../../Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values
    eye_df_native = eyemovement_df[eyemovement_df['list'].isin(reader_list)].reset_index(drop=True)
    word_info_df = word_info_df[word_info_df['list'].isin(reader_list)].reset_index(drop=True)

    # For duration prediction determine the min and max values
    # Calculate IQR for the 'dur' column
    Q1 = eye_df_native["CURRENT_FIX_DURATION"].quantile(0.25)
    Q3 = eye_df_native["CURRENT_FIX_DURATION"].quantile(0.75)
    IQR = Q3 - Q1

    # Determine bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter durations within the bounds
    filtered_durations = eye_df_native["CURRENT_FIX_DURATION"][
        (eye_df_native["CURRENT_FIX_DURATION"] >= lower_bound) & (eye_df_native["CURRENT_FIX_DURATION"] <= upper_bound)]

    # Determine min_dur and max_dur
    min_dur = filtered_durations.min()
    max_dur = filtered_durations.max()

    all_subjects_df = pd.DataFrame()
    # notice: Each sentence is recorded multiple times in file |word_info_df|.
    grouped_df = word_info_df.groupby(['sentenceid', 'WORD_NORM'])  # 5456 unique sentences
    word_info_df = grouped_df.head(1)  # remove repeating sent ids
    word_info_df = word_info_df[['list', 'sentenceid', 'WORD_NORM']]
    for subject_id in range(1, len(reader_list) + 1):  # 69 subjects (native speakers) in the CELER dataset
        CELER_pred = uniform_fixation_model("CELER", word_info_df.copy(), min_dur, max_dur)
        CELER_pred['subj_id'] = subject_id
        print(CELER_pred)
        all_subjects_df = all_subjects_df.append(CELER_pred)
    all_subjects_df.to_csv('CELER_uniform_results.csv', index=False)
    return all_subjects_df


def map_sentence_name(row, sent_dict):
    original_name = row['SN']
    mapped_name = sent_dict.get(original_name)
    return mapped_name


def evaluate_uniform_model(dataset, sp_human, landing_pos_mean, landing_pos_std, fix_dur_mean, fix_dur_std,
                           landing_pos_mean_uniform, landing_pos_std_uniform, fix_dur_mean_uniform,
                           fix_dur_std_uniform):
    if dataset == "BSC":
        results = pd.read_csv("BSC_uniform_results.csv")
    else:  # celer
        results = pd.read_csv("CELER_uniform_results.csv")
        with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)
            results['SN'] = results.apply(map_sentence_name, axis=1, args=(sent_dict,))

    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sp_human["sent_id"])

    sampled_dfs = []
    for sent in set(sp_human["sent_id"]):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(results, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    # create a scanpath dict based on the results df
    scanpaths = create_sp(dataset, sp_df)

    # calculate central scasim scores
    if dataset == "BSC":
        central_sp_path = "../../BSC_most_central_sp.txt"
        central_scasim_scores = compute_central_scasim(central_sp_path, scanpaths)
    else:  # CELER
        central_sp_path = "../../CELER_most_central_sp.txt"
        central_scasim_scores = compute_central_scasim(central_sp_path, scanpaths,
                                                       '../../data_splits/CELER_sent_dict.txt')

    # calculate "normal" scasim scores
    scasim_scores = compute_scasim(scanpaths, sp_human)

    # calculate MSE scores for durations
    dur_mse_scores = []
    max_len_predicted = max(len(sp) for sp in scanpaths["durations"])
    max_len_target = max(len(sp) for sp in sp_human["durations"])
    max_len = max(max_len_predicted, max_len_target)
    criterion = torch.nn.MSELoss(reduction='mean')
    for predicted_sp, target_sp in zip(scanpaths["durations"], sp_human["durations"]):
        # Pad shorter sequences
        padding_len_predicted = max_len - len(predicted_sp)
        predicted_sp_padded = predicted_sp + [0] * padding_len_predicted
        padding_len_target = max_len - len(target_sp)
        target_sp_padded = target_sp + [0] * padding_len_target

        predicted_sp_padded = torch.tensor(predicted_sp_padded)
        target_sp_padded = torch.tensor(target_sp_padded)

        pad_mask = ~(target_sp_padded == 0)
        predicted_sp_padded = predicted_sp_padded * pad_mask
        target_sp_padded = target_sp_padded * pad_mask

        # apply z-score normalisation
        predicted_sp_padded = predicted_sp_padded / 1000
        target_sp_padded = target_sp_padded / 1000
        predicted_sp_padded = (predicted_sp_padded - fix_dur_mean_uniform) / fix_dur_std_uniform * pad_mask
        target_sp_padded = (target_sp_padded - fix_dur_mean) / fix_dur_std * pad_mask

        dur_mse = criterion(predicted_sp_padded[pad_mask], target_sp_padded[pad_mask])
        dur_mse_scores.append(dur_mse.item())

    # calculate MSE scores for landing pos
    land_pos_mse_scores = []
    max_len_predicted = max(len(sp) for sp in scanpaths["landing_pos"])
    max_len_target = max(len(sp) for sp in sp_human["landing_pos"])
    max_len = max(max_len_predicted, max_len_target)
    criterion = torch.nn.MSELoss(reduction='mean')
    for predicted_sp, target_sp in zip(scanpaths["landing_pos"], sp_human["landing_pos"]):
        # Pad shorter sequences
        padding_len_predicted = max_len - len(predicted_sp)
        predicted_sp_padded = predicted_sp + [0] * padding_len_predicted
        padding_len_target = max_len - len(target_sp)
        target_sp_padded = target_sp + [0] * padding_len_target

        predicted_sp_padded = torch.tensor(predicted_sp_padded)
        target_sp_padded = torch.tensor(target_sp_padded)

        pad_mask = ~(target_sp_padded == 0)
        predicted_sp_padded = predicted_sp_padded * pad_mask
        target_sp_padded = target_sp_padded * pad_mask

        # apply z-score normalisation
        predicted_sp_padded = (predicted_sp_padded - landing_pos_mean_uniform) / landing_pos_std_uniform * pad_mask
        target_sp_padded = (target_sp_padded - landing_pos_mean) / landing_pos_std * pad_mask

        land_pos_mse = criterion(predicted_sp_padded[pad_mask], target_sp_padded[pad_mask])
        land_pos_mse_scores.append(land_pos_mse.item())

    return central_scasim_scores, scasim_scores, dur_mse_scores, land_pos_mse_scores


def compute_mean_std_uniform(simulation_results):
    df = pd.read_csv(simulation_results)
    mean_dur = df['dur'].mean()
    sd_dur = df['dur'].std()

    mean_land_pos = df['land_pos'].mean()
    sd_land_pos = df['land_pos'].std()

    return mean_dur, sd_dur, mean_land_pos, sd_land_pos


def construct_uniform_tensor(location_preds_test):
    batch_size, seq_len, step_size = location_preds_test.shape
    uniform_tensor = torch.full((batch_size, seq_len, step_size), 1 / step_size)
    return uniform_tensor.numpy()
