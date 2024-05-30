from utils import *
from collections import Counter
import random


def parse_line(line, subj, previous_sent):
    """
        Parses the E-Z Reader simulation results file line by line.

        Args:
            line (str): A line from the simulation results file.
            subj (int): Current subject ID.
            previous_sent (int): ID of the previous sentence.

        Returns:
            tuple: A tuple containing (duration, word, location, sentence_id, subject_id).
        """
    parts = line.split()

    # Extract the duration
    dur_index = parts.index('dur:') + 1
    duration = int(parts[dur_index])

    # Extract the location
    word_index = parts.index('word:') + 1
    location = int(parts[word_index]) + 1

    # Extract the sentence id
    sent_index = parts.index('sent:') + 1
    sent = int(parts[sent_index])

    # Extract the actual word
    word = ' '.join(parts[sent_index + 1:])
    word = word.replace('.', '')

    # Update subject for new sentence
    subj = subj + 1 if sent != previous_sent else subj

    return duration, word, location, sent, subj


def create_sp(df):
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
    grouped_data = df.groupby('subj')

    locations = []
    durations = []
    sent_id = []

    for _, sentence_data in grouped_data:
        locations.append(sentence_data['loc'].to_numpy().astype(float))
        durations.append(sentence_data['dur'].tolist())
        sent_id.append(sentence_data['sent'].iloc[0])

    scanpath = {
        'locations': locations,
        'durations': durations,
        'sent_id': sent_id
    }

    for i, loc_array in enumerate(scanpath["locations"]):
        loc_list = loc_array.tolist()
        loc_list.insert(0, 0)
        loc_list.append(26)
        scanpath["locations"][i] = np.array(loc_list)

    for i, dur_array in enumerate(scanpath["durations"]):
        dur_array.insert(0, -0.0)
        dur_array.append(-0.0)
        scanpath["durations"][i] = dur_array

    return scanpath


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
    sentence_df = df[df['sent'] == sent]

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
        random_sp = random.sample(sentence_df['subj'].tolist(), actual_count)
        result_df = sentence_df[sentence_df['subj'].isin(random_sp)]
    else:
        # Use all available scanpaths
        result_df = sentence_df.copy()

    return result_df


def main(dataset, dataset_path, simulation_results, data_split):
    # Parse the simulation output of E-Z Reader
    data = []
    subj = 1
    previous_sent = 1
    with open(simulation_results, 'r',
              encoding='utf-8') as file:
        for line in file:
            duration, word, location, sent, subj = parse_line(line, subj, previous_sent)
            data.append((duration, word, location, sent, subj))
            previous_sent = sent

    df = pd.DataFrame(data, columns=['dur', 'word', 'loc', 'sent', 'subj'])  # simulation results df

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
        sampled_sp = sample_scanpaths(df, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    # format the scanpath in the correct format
    scanpath = create_sp(sp_df)

    sn_ids_test = torch.tensor(sn_ids_test)
    # Compute the Scasim scores
    scasim_scores = compute_scasim(dataset, dataset_path, scanpath, sn_ids_test)
    return scasim_scores, np.mean(scasim_scores)


if __name__ == "__main__":
    BSC_results_df = pd.DataFrame(columns=['dataset', 'scasim_scores', 'mean_scasim'])
    data_split_path = '../../data_splits/'
    # Evaluate E-Z Reader on BSC (New Sentence, New Reader, NRS)
    BSC_path = '../../Data/beijing-sentence-corpus/'
    BSC_sim_results = 'BSCSimulationResults.txt'
    BSC_splits = ["BSC_new_sentence_splits", "BSC_new_subject_splits", "BSC_NRS_split"]
    for dataset in BSC_splits:
        scasim_scores, mean_score = main("BSC", BSC_path, BSC_sim_results,
                                        os.path.join(data_split_path, f'BSC/{dataset}.txt'))
    BSC_results_df = BSC_results_df.append({'dataset': dataset, 'scasim_scores': scasim_scores, 'mean_scasim': mean_score},
                                    ignore_index=True)
    BSC_results_df.to_csv("BSC_scasim_results.csv", index=False)

    # Evaluate E-Z Reader on CELER (New Sentence, New Reader, NRS)
    CELER_results_df = pd.DataFrame(columns=['dataset', 'scasim_scores', 'mean_scasim'])
    CELER_path = '../../Data/celer/'
    CELER_sim_results = 'CELERSimulationResults.txt'
    CELER_splits = ["CELER_new_reader_splits", "CELER_new_sentence_splits", "CELER_NRS_split"]
    for dataset in CELER_splits:
        scasim_scores, mean_score = main("CELER", CELER_path, CELER_sim_results,
                                         os.path.join(data_split_path, f'CELER/{dataset}.txt'))
        CELER_results_df = CELER_results_df.append(
            {'dataset': dataset, 'scasim_scores': scasim_scores, 'mean_scasim': mean_score},
            ignore_index=True)
    CELER_results_df.to_csv("CELER_scasim_results.csv", index=False)
