from scasim import *
from collections import Counter


def evaluate_swift(dataset, sp_human, fix_dur_mean_uniform, fix_dur_std_uniform, fix_dur_mean, fix_dur_std,
                   landing_pos_mean, landing_pos_std, landing_pos_mean_uniform, landing_pos_std_uniform):
    df = pd.read_csv("drive/MyDrive/baseline/swift/fixseqin_generation_cleaned.csv", delimiter='\t')

    # Count sentence occurrences in the test splits
    sentence_counts = Counter(sp_human["sent_id"])

    sampled_dfs = []
    for sent in set(sp_human["sent_id"]):
        count = sentence_counts[sent]
        sampled_sp = sample_scanpaths(df, sent, count)
        sampled_dfs.append(sampled_sp)

    sp_df = pd.concat(sampled_dfs, ignore_index=True)

    # create a scanpath dict based on the results df
    scanpaths = create_sp(dataset, sp_df)

    # calculate central scasim scores
    if dataset == "celer":
        central_sp_path = "drive/MyDrive/CELER_most_central_sp.txt"  # ../..
        central_scasim_scores = compute_central_scasim(central_sp_path, scanpaths,
                                                       'drive/MyDrive/CELER_sent_dict.txt')  # ../../data_splits/
    # calculate "normal" scasim scores
    scasim_scores = compute_scasim(scanpaths, sp_human)

    # calculate MSE scores for durations
    dur_mse_scores = []
    max_len_predicted = max(len(sp) for sp in scanpaths["durations"])
    max_len_target = max(len(sp) for sp in sp_human["durations"])
    max_len = max(max_len_predicted, max_len_target)
    criterion = torch.nn.MSELoss(reduction='mean')
    for predicted_sp, target_sp in zip(scanpaths["durations"], sp_human["durations"]):
        # Pad shorter sequences with a chosen padding value (e.g., 0)
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

    # calculate NLL scores for locations
    nll_scores = []
    max_len_predicted = max(len(sp) for sp in scanpaths["locations"])
    max_len_target = max(len(sp) for sp in sp_human["locations"])
    max_len = max(max_len_predicted, max_len_target)
    for predicted_sp, target_sp in zip(scanpaths["locations"], sp_human["locations"]):
        # Pad shorter sequences with a chosen padding value (e.g., 0)
        padding_len_predicted = max_len - len(predicted_sp)
        predicted_sp_padded = predicted_sp.tolist() + [0] * padding_len_predicted
        padding_len_target = max_len - len(target_sp)
        target_sp_padded = target_sp + [0] * padding_len_target

        predicted_sp_padded = torch.tensor(predicted_sp_padded)
        target_sp_padded = torch.tensor(target_sp_padded)

        pad_mask = ~(target_sp_padded == 0)
        predicted_sp_padded = predicted_sp_padded * pad_mask
        target_sp_padded = target_sp_padded * pad_mask

        nll_score = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(predicted_sp_padded[pad_mask], dim=0), target_sp_padded[pad_mask],
            ignore_index=0)
        nll_scores.append(nll_score.item())

    return central_scasim_scores, scasim_scores, dur_mse_scores, land_pos_mse_scores, nll_scores


def compute_mean_std_swift():
    df = pd.read_csv("drive/MyDrive/baseline/swift/fixseqin_generation_cleaned.csv", delimiter='\t')
    mean_dur = df['dur'].mean()
    sd_dur = df['dur'].std()

    mean_land_pos = df['landing_pos'].mean()
    sd_land_pos = df['landing_pos'].std()

    return mean_dur, sd_dur, mean_land_pos, sd_land_pos


