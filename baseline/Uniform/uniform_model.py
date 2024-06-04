from collections import Counter
from scasim import *
import pandas as pd
import random


def sample_scanpaths(df, sent, count):
    """
  Samples scanpaths for a given sentence with a specified count.

  Args:
      df (pd.DataFrame): Dataframe containing scanpath data ('dur', 'word', 'loc', 'sent', 'subj').
      sent (int): The sentence ID for which to sample scanpaths.
      count (int): The number of scanpaths to sample.

  Returns:
      pd.dataframe: A DataFrame containing the sampled (or all) scanpaths
                   for the specified sentence.
  """
    # Filter scanpaths for the current sentence
    sentence_df = df[df['SN'] == sent]

    # Get the number of available scanpaths for this sentence
    available_count = len(sentence_df)

    # Determine the actual number of scanpaths to return
    if count > available_count:
        print(
            f"Requested count ({count}) exceeds available scanpaths ({available_count}) for sentence {sent}. Using all available scanpaths.")
        actual_count = available_count
    else:
        actual_count = count

    # Randomly sample n scanpaths from the available ones (if applicable)
    if actual_count < available_count:
        random_sp = random.sample(sentence_df['subj_id'].tolist(), actual_count)
        result_df = sentence_df[sentence_df['subj_id'].isin(random_sp)]
    else:
        # Use all available scanpaths
        result_df = sentence_df.copy()

    return result_df


def uniform_fixation_model(dataset, df, min_dur, max_dur):
    """
  This function predicts fixation locations uniformly across a sentence.

  Args:
      df (pd.DataFrame): A pandas dataframe with columns 'WORD' and 'SN'

  Returns:
      pd.DataFrame: A dataframe with additional columns 'X', 'Y' (fixation coordinates)
  """
    # Initialize an empty DataFrame to store sentence-level data
    sentence_df = pd.DataFrame(columns=['SN', 'WORD', 'loc', 'dur'])
    if dataset == "BSC":
        grouped_df = df.groupby('SN')['WORD'].apply(list)
    else:  # CELER
        grouped_df = df.groupby('sentenceid')['WORD_NORM'].apply(list)
    for sentence_num, sentence_words in grouped_df.items():
        num_words = len(sentence_words)
        # Generate random indices (locations) for each word in the sentence
        fixation_indices = np.random.randint(1, num_words + 1, size=num_words)
        # Generate random durations
        durations = np.random.randint(min_dur, max_dur + 1, size=num_words)
        sentence_data = pd.DataFrame(
            {'SN': sentence_num, 'WORD': sentence_words, 'loc': fixation_indices, 'dur': durations})
        sentence_df = sentence_df.append(sentence_data)

    sentence_df = sentence_df.reset_index(drop=True)
    return sentence_df


def compute_uniform_bsc():
    bsc_path = '.././Data/beijing-sentence-corpus/'
    info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
    bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')
    eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')

    # For duration prediction determine the min and max values based on the Q1 and Q3
    # Calculate quartiles of the 'dur' column
    q1 = eyemovement_df['dur'].quantile(0.25)
    q3 = eyemovement_df['dur'].quantile(0.75)
    # Filter data within the interquartile range (IQR)
    filtered_df = eyemovement_df[(eyemovement_df['dur'] >= q1) & (eyemovement_df['dur'] <= q3)]
    # Find the maximum and minimum values within the filtered data
    max_dur = filtered_df['dur'].max()
    min_dur = filtered_df['dur'].min()

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
    eyemovement_df = pd.read_csv('.././Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace(
        '\t(.*)', '', regex=True)
    word_info_df = pd.read_csv('.././Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # native speakers only
    sub_metadata_path = '.././Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values
    eye_df_native = eyemovement_df[eyemovement_df['list'].isin(reader_list)].reset_index(drop=True)
    word_info_df = word_info_df[word_info_df['list'].isin(reader_list)].reset_index(drop=True)

    # For duration prediction determine the min and max values based on the Q1 and Q3
    # Calculate quartiles of the 'dur' column
    q1 = eye_df_native['CURRENT_FIX_DURATION'].quantile(0.25)
    q3 = eye_df_native['CURRENT_FIX_DURATION'].quantile(0.75)
    # Filter data within the interquartile range (IQR)
    filtered_df = eye_df_native[
        (eye_df_native['CURRENT_FIX_DURATION'] >= q1) & (eye_df_native['CURRENT_FIX_DURATION'] <= q3)]
    # Find the maximum and minimum values within the filtered data
    max_dur = filtered_df['CURRENT_FIX_DURATION'].max()
    min_dur = filtered_df['CURRENT_FIX_DURATION'].min()

    all_subjects_df = pd.DataFrame()
    # notice: Each sentence is recorded multiple times in file |word_info_df|.
    grouped_df = word_info_df.groupby(['sentenceid', 'WORD_NORM'])  # 5456 unique sentences
    word_info_df = grouped_df.head(1)  # remove repeating sent ids
    word_info_df = word_info_df[['list', 'sentenceid', 'WORD_NORM']]
    for subject_id in range(1, len(reader_list) + 1):  # 69 subjects (native speakers) in the CELER dataset
        CELER_pred = uniform_fixation_model("CELER", word_info_df.copy(), min_dur, max_dur)
        CELER_pred['subj_id'] = subject_id
        all_subjects_df = all_subjects_df.append(CELER_pred)
    all_subjects_df.to_csv('CELER_uniform_results.csv', index=False)
    return all_subjects_df


def create_sp(dataset, df):
    """
        Creates a scanpath dictionary from the provided DataFrame with the following structure:

        scanpath = {
            'locations': [locations_list1, locations_list2, ...],  # List of location arrays for each sentence
            'durations': [durations_list1, durations_list2, ...],  # List of duration lists for each sentence
            'sent_id': [sent_id1, sent_id2, ...]                    # List of sentence IDs
        }

        Args:
            df (pd.DataFrame): DataFrame containing scanpath data ('dur', 'word', 'loc', 'sent', 'subj').

        Returns:
            dict: The scanpath dictionary with locations, durations, and sentence IDs.
        """
    grouped_data = df.groupby(['subj_id', 'SN'])  # maybe group by subj and sn

    locations = []
    durations = []
    sent_id = []

    for _, subj_data in grouped_data:
        locations.append(subj_data['loc'].to_numpy().astype(float))  # len(locations) = 9000
        durations.append(subj_data['dur'].to_numpy().astype(int))  # len(durations) = 9000
        sent_id.append(subj_data['SN'].iloc[0])  # 9000

    scanpath = {
        'locations': locations,
        'durations': durations,
        'sent_id': sent_id
    }

    for i, loc_array in enumerate(scanpath["locations"]):
        loc_list = loc_array.tolist()
        loc_list.insert(0, 0)
        if dataset == "BSC":
            loc_list.append(26)
        else:  # CELER
            loc_list.append(23)
        scanpath["locations"][i] = np.array(loc_list)

    for i, dur_array in enumerate(scanpath["durations"]):
        dur_list = dur_array.tolist()
        dur_list.insert(0, -0.0)
        dur_list.append(-0.0)
        scanpath["durations"][i] = dur_list
    return scanpath


def map_sentence_name(row, sent_dict):
    original_name = row['SN']
    mapped_name = sent_dict.get(original_name)
    return mapped_name


def evaluate_uniform_model(dataset, data_split):
    if dataset == "BSC":
        results = pd.read_csv("BSC_uniform_results.csv")
    else:  # CELER
        results = pd.read_csv("CELER_uniform_results.csv")
        with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)
        results['SN'] = results.apply(map_sentence_name, axis=1, args=(sent_dict,))

    # retrieve the original Eyettention test splits
    sn_ids_test = []
    with open(data_split, 'r') as file:
        for line in file:
            for number in line.strip().split(","):
                number = number.replace(" ", "")
                if number != "":
                    sn_ids_test.append(int(number))
    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sn_ids_test)

    sampled_dfs = []
    for sent in set(sn_ids_test):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(results, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    # create a scanpath dict based on the results df
    scanpaths = create_sp(dataset, sp_df)

    # load the most central scanpaths
    if dataset == "BSC":
        central_sp_path = "../../BSC_most_central_sp.txt"
        scasim_scores = compute_scasim(central_sp_path, scanpaths)
    else:  # CELER
        central_sp_path = "../../CELER_most_central_sp.txt"
        scasim_scores = compute_scasim(central_sp_path, scanpaths, '../../data_splits/CELER_sent_dict.txt')

    print("Mean Scasim score", np.mean(scasim_scores))


if __name__ == "__main__":
    compute_uniform_bsc()
    compute_uniform_celer()
    evaluate_uniform_model("BSC", '../../data_splits/BSC/BSC_new_sentence_splits.txt')
    evaluate_uniform_model("BSC", '../../data_splits/BSC/BSC_new_subject_splits.txt')
    evaluate_uniform_model("BSC", '../../data_splits/BSC/BSC_NRS_split.txt')
    evaluate_uniform_model("CELER", '../../data_splits/CELER/CELER_new_sentence_splits.txt')
    evaluate_uniform_model("CELER", '../../data_splits/CELER/CELER_new_reader_splits.txt')
    evaluate_uniform_model("CELER", '../../data_splits/CELER/CELER_NRS_split.txt')
