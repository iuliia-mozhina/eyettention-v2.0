from scasim import *
from utils import *


def parse_line(line):
    parts = line.split()
    if len(parts) != 9:
        return None
    else:
        # Extract the duration
        dur_index = parts.index('dur:') + 1
        duration_str = parts[dur_index]
        if duration_str == 'Infinity':
            duration = 0
        else:
            duration = int(duration_str)

        # Extract the landing pos
        land_pos_index = parts.index('pos:') + 1
        land_pos = str(parts[land_pos_index])
        land_pos = land_pos.replace(",", ".")
        land_pos = float(land_pos)

        # Extract the location
        word_index = parts.index('word:') + 1
        location = int(parts[word_index]) + 1

        # Extract the sentence id
        sent_index = parts.index('sent:') + 1
        sent = int(parts[sent_index])

        # Extract the actual word
        word = ' '.join(parts[sent_index + 1:])
        word = word.replace('.', '')

    return duration, land_pos, word, location, sent


def evaluate_ez_reader(dataset, simulation_results, sp_human, landing_pos_mean, landing_pos_std, fix_dur_mean,
                       fix_dur_std,
                       landing_pos_mean_ez_reader, landing_pos_std_ez_reader, fix_dur_mean_ez_reader,
                       fix_dur_std_ez_reader):
    # Parse the simulation output of E-Z Reader
    data = []
    with open(simulation_results, 'r',
              encoding='utf-8') as file:
        for line in file:
            parsed_data = parse_line(line)
            if parsed_data is not None:
                duration, land_pos, word, location, sent = parsed_data
                data.append((duration, land_pos, word, location, sent))

    df = pd.DataFrame(data, columns=['dur', 'land_pos', 'word', 'loc', 'SN'])

    # Add a subject column
    df['subj_id'] = ((df['SN'] == 0) & (df['SN'].shift(1) != 0)).cumsum()

    # convert the landing pos to landing pos relatively to the word beginning
    sp_fix_loc = df.land_pos.values
    sp_landing_pos_char = np.modf(sp_fix_loc)[0]
    df["land_pos"] = sp_landing_pos_char

    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sp_human["sent_id"])

    sampled_dfs = []
    for sent in set(sp_human["sent_id"]):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(df, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    # format the scanpath in the correct format
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
        predicted_sp_padded = (predicted_sp_padded - fix_dur_mean_ez_reader) / fix_dur_std_ez_reader * pad_mask
        target_sp_padded = (target_sp_padded - fix_dur_mean) / fix_dur_std * pad_mask

        dur_mse = criterion(predicted_sp_padded[pad_mask], target_sp_padded[pad_mask])
        dur_mse_scores.append(dur_mse.item())

    # calculate MSE scores for landing pos
    land_pos_mse_scores = []
    max_len_predicted = max(len(sp) for sp in scanpaths["landing_pos"]) - 1
    max_len_target = max(len(sp) for sp in sp_human["landing_pos"])
    max_len = max(max_len_predicted, max_len_target)
    criterion = torch.nn.MSELoss(reduction='mean')
    for predicted_sp, target_sp in zip(scanpaths["landing_pos"], sp_human["landing_pos"]):
        predicted_sp = predicted_sp[:1] + predicted_sp[2:]
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
        predicted_sp_padded = (predicted_sp_padded - landing_pos_mean_ez_reader) / landing_pos_std_ez_reader * pad_mask
        target_sp_padded = (target_sp_padded - landing_pos_mean) / landing_pos_std * pad_mask

        land_pos_mse = criterion(predicted_sp_padded[pad_mask], target_sp_padded[pad_mask])
        land_pos_mse_scores.append(land_pos_mse.item())

    return central_scasim_scores, scasim_scores, dur_mse_scores, land_pos_mse_scores


def compute_mean_std_ez_reader(simulation_results):
    # Parse the simulation output of E-Z Reader
    data = []
    with open(simulation_results, 'r',
              encoding='utf-8') as file:
        for line in file:
            parsed_data = parse_line(line)
            if parsed_data is not None:
                duration, land_pos, word, location, sent = parsed_data
                data.append((duration, land_pos, word, location, sent))

    df = pd.DataFrame(data, columns=['dur', 'land_pos', 'word', 'loc', 'SN'])

    # convert the landing pos to landing pos relatively to the word beginning
    sp_fix_loc = df.land_pos.values
    sp_landing_pos_char = np.modf(sp_fix_loc)[0]
    df["land_pos"] = sp_landing_pos_char

    mean_dur = df['dur'].mean()
    sd_dur = df['dur'].std()

    mean_land_pos = df['land_pos'].mean()
    sd_land_pos = df['land_pos'].std()

    return mean_dur, sd_dur, mean_land_pos, sd_land_pos

