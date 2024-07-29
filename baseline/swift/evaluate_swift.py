from scasim import *
from collections import Counter


def evaluate_swift(dataset, sp_human, fix_dur_mean_uniform, fix_dur_std_uniform, fix_dur_mean, fix_dur_std,
                   landing_pos_mean, landing_pos_std, landing_pos_mean_uniform, landing_pos_std_uniform):
    df = pd.read_csv("fixseqin_generation_cleaned.csv", delimiter='\t')

    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sp_human["sent_id"])

    sampled_dfs = []
    for sent in set(sp_human["sent_id"]):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(df, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    scanpaths = create_sp(dataset, sp_df)

    # calculate central scasim scores
    if dataset == "celer":
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


def compute_mean_std_swift():
    df = pd.read_csv("baseline/swift/fixseqin_generation_cleaned.csv", delimiter='\t')
    mean_dur = df['dur'].mean()
    sd_dur = df['dur'].std()

    mean_land_pos = df['land_pos'].mean()
    sd_land_pos = df['land_pos'].std()

    return mean_dur, sd_dur, mean_land_pos, sd_land_pos


def evaluate_swift_nll(nll_file, sp_human):
    df = pd.read_excel(nll_file)
    df['SN'] = df['i'].str.extract(r'S(\d+)R')

    # generate subject id
    def custom_index(sents):
        index = 1
        result = [index]
        for i in range(1, len(sents)):
            if sents[i] == sents[i - 1]:
                index += 1
            else:
                index = 1
            result.append(index)
        return result

    df['subj_id'] = custom_index(df['SN'].tolist())

    df['SN'] = df['SN'].astype(int)
    df['nll'] = df['nll'].astype(int).astype(float)

    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sp_human["sent_id"])

    sampled_dfs = []
    for sent in set(sp_human["sent_id"]):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(df, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    if not sp_df.empty:
        nll_values = sp_df['nll'].tolist()
        return nll_values
    else:
        return []

