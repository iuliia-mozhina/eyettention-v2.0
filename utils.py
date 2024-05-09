#coding=utf-8
import numpy as np
import pandas as pd
import os
from typing import Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from math import acos, cos, sin, pi

def load_bsc() -> Tuple[pd.DataFrame, ...]:
    """
    :return: word info dataframe, part-of-speech info, eye movements
    """
    bsc_path = './Data/beijing-sentence-corpus/'
    info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
    bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')
    word_info_df = pd.read_excel(info_path, 'word')
    pos_info_df = pd.read_excel(info_path, header=None)
    eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
    return word_info_df, pos_info_df, eyemovement_df

def load_corpus(corpus, task=None):
    if corpus == 'BSC':
        #load word data, POS data, EM data
        word_info_df, pos_info_df, eyemovement_df = load_bsc()
        return word_info_df, pos_info_df, eyemovement_df
    elif corpus == 'celer':
        eyemovement_df = pd.read_csv('./Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
        eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)', '', regex=True)
        word_info_df = pd.read_csv('./Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
        word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
        return word_info_df, None, eyemovement_df

def compute_BSC_word_length(sn_df):
    word_len = sn_df.LEN.values
    wl_list = []
    for wl in word_len:
        wl_list.extend([wl]*wl)
    arr = np.asarray(wl_list, dtype=np.float32)
    #length of a punctuation is 0, plus an epsilon to avoid division output inf
    arr[arr==0] = 1/(0+0.5)
    arr[arr!=0] = 1/(arr[arr!=0])
    return arr

def pad_seq(seqs, max_len, pad_value, dtype=np.compat.long):
    padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
    for i, seq in enumerate(seqs):
        padded[i, 0] = 0
        padded[i, 1:(len(seq)+1)] = seq
        if pad_value !=0:
            padded[i, len(seq)+1] = pad_value -1

    return padded

def pad_seq_with_nan(seqs, max_len, dtype=np.compat.long):
    padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
    for i, seq in enumerate(seqs):
        padded[i, 1:(len(seq)+1)] = seq
    return padded

def _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
    """
    SN_token_embedding   <CLS>, bla, bla, <SEP>
    SP_token_embedding   <CLS>, bla, bla, <SEP>
    SP_ordinal_pos 0, bla, bla, max_sp_len
    SP_fix_dur     0, bla, bla, 0
    SN_len         original sentence length without start and end tokens
    """
    SN_ids, SN_pred, SN_word_freq = [], [], []
    SN_input_ids, SN_attention_mask, SN_WORD_len = [], [], []
    SP_input_ids, SP_attention_mask = [], []
    SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
    sub_id_list = []
    for sn_id in sn_list:
        #process sentence sequence
        sn_df = eyemovement_df[eyemovement_df.sn==sn_id]
        sn = word_info_df[word_info_df.SN == sn_id]
        sn_str = ''.join(sn.WORD.values)
        sn_word_len = compute_BSC_word_length(sn)
        predictability = sn.PRED.values
        word_freq = sn.WF_BLI.values

        #tokenization and padding
        tokenizer.padding_side = 'right'
        tokens = tokenizer.encode_plus(sn_str,
                                        add_special_tokens = True,
                                        truncation=True,
                                        max_length = cf["max_sn_len"],
                                        padding = 'max_length',
                                        return_attention_mask=True)
        encoded_sn = tokens["input_ids"]
        mask_sn = tokens["attention_mask"]

        #process fixation sequence
        for sub_id in reader_list:
            sub_df = sn_df[sn_df.id==sub_id]
            if len(sub_df) == 0:
                #no scanpath data found for the subject
                continue

            #last fixation go back to the first character with fl = 0 -- seems to be outlier point? remove
            if sub_df.iloc[-1].wn == 1 and sub_df.iloc[-1].fl == 0:
                sub_df = sub_df.iloc[:-1]

            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values
            sp_landing_pos_char = np.modf(sp_fix_loc)[0]
            SP_landing_pos.append(sp_landing_pos_char)

            #Convert word-based ordinal positions to token(character)-based ordinal positions
            #When the fixated word index is less than 0, set it to 0
            sp_fix_loc = np.where(sp_fix_loc<0, 0, sp_fix_loc)
            sp_ordinal_pos = [np.sum(sn[sn.NW<value].LEN) + np.ceil(sp_fix_loc[count]+ 1e-10) for count, value in enumerate(sp_word_pos)]
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)

            #tokenization and padding for scanpath, i.e. fixated word sequence
            sp_token = [sn_str[int(i-1)] for i in sp_ordinal_pos]
            sp_token_str = '[CLS]' + ''.join(sp_token) + '[SEP]'
            sp_tokens = tokenizer.encode_plus(sp_token_str,
                                                add_special_tokens = False,
                                                truncation=True,
                                                max_length = cf["max_sp_len"],
                                                padding = 'max_length',
                                                return_attention_mask=True)
            encoded_sp = sp_tokens["input_ids"]
            mask_sp = sp_tokens["attention_mask"]
            SP_input_ids.append(encoded_sp)
            SP_attention_mask.append(mask_sp)

            #sentence information
            SN_input_ids.append(encoded_sn)
            SN_attention_mask.append(mask_sn)
            SN_WORD_len.append(sn_word_len)
            sub_id_list.append(sub_id)
            SN_ids.append(sn_id)
            SN_pred.append(predictability)
            SN_word_freq.append(word_freq)

    #padding for batch computation
    SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
    SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
    SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
    SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)
    SN_pred = pad_seq_with_nan(SN_pred, cf["max_sn_len"], dtype=np.float32)
    SN_word_freq = pad_seq_with_nan(SN_word_freq, cf["max_sn_len"], dtype=np.float32)

    #assign type
    SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
    SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
    SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
    SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
    sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
    SN_ids = np.asarray(SN_ids, dtype=np.int64)

    data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len,
            "SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask,
            "SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
            "sub_id": sub_id_list, "SN_ids": SN_ids, "SN_pred": SN_pred, "SN_word_freq": SN_word_freq}

    return data

class BSCdataset(Dataset):
    """Return BSC dataset."""

    def __init__(
        self,
        word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
    ):
        self.data = _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

    def __len__(self):
        return len(self.data["SN_input_ids"])

    def __getitem__(self,idx):
        sample = {}
        sample["sn_ids"] = self.data["SN_ids"][idx]
        sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
        sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
        sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
        sample["sn_pred"] = self.data['SN_pred'][idx, :]
        sample["sn_word_freq"] = self.data['SN_word_freq'][idx, :]

        sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
        sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]

        sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
        sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]

        sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

        sample["sub_id"] = self.data["sub_id"][idx]

        return sample

def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
    #calculate mean
    total_sum = 0
    total_num = 0
    for batchh in dataloader:
        batchh.keys()
        feat = batchh[feat_key]/scale
        feat = torch.nan_to_num(feat)
        total_num += len(feat.view(-1).nonzero())
        total_sum += feat.sum()
    feat_mean = total_sum / total_num
    #calculate std
    sum_of_squared_error = 0
    for batchh in dataloader:
        batchh.keys()
        feat = batchh[feat_key]/scale
        feat = torch.nan_to_num(feat)
        mask = ~torch.eq(feat, padding_value)
        sum_of_squared_error += (((feat - feat_mean).pow(2))*mask).sum()
    feat_std = torch.sqrt(sum_of_squared_error / total_num)
    return feat_mean, feat_std

def load_label(sp_pos, cf, labelencoder, device):
    #prepare label and mask
    pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
    end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
    mask = pad_mask + end_mask
    sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
    label = sp_pos[:, 1:]*mask + sac_amp*~mask
    label = torch.where(label>cf["max_sn_len"]-1, cf["max_sn_len"]-1, label).to('cpu').detach().numpy()
    label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
    if device == 'cpu':
        pad_mask = pad_mask.to('cpu').detach().numpy()
    else:
        label = torch.from_numpy(label).to(device)
    return pad_mask, label


def likelihood(pred, label, mask):
    #test
    #res = F.nll_loss(torch.tensor(pred), torch.tensor(label))
    label = one_hot_encode(label, pred.shape[1])
    res = np.sum(np.multiply(pred, label), axis=1)
    res = np.sum(res * ~mask)/np.sum(~mask)
    return res

def eval_log_llh(dnn_out, label, pad_mask):
    res = []
    dnn_out = np.log2(dnn_out + 1e-10)
    #For each scanpath calculate the likelihood and then find the average
    for sp_indx in range(dnn_out.shape[0]):
        out = likelihood(dnn_out[sp_indx, :, :], label[sp_indx, :], pad_mask[sp_indx, :])
        res.append(out)

    return res


def which_min(*l):
    mi = 0
    for i,e in enumerate(l):
        if e<l[mi]:
            mi = i
    return mi


def _compute_central_sp(dataset, sn_id):
    """
    Calculates the most central scanpath for a given sentence in the dataset.
    The most central scanpath is the sp with the smallest average dissimilarity from every other sp for a given sentence.
    """
    _, _, eyemovement_df = load_corpus(dataset)
    # Filter eyemovement data for the given sentence
    eyemovement_df_sn = eyemovement_df[eyemovement_df['sn'] == sn_id]

    grouped_scanpaths = []
    subject_ids = []
    for subject_id, subject_data in eyemovement_df_sn.groupby('id'): # Group by subject ID and prepare scanpaths in the correct format (x-coordinate, y-coordinate, duration)
    # Convert fixation data into (x, y, duration) tuples
        scanpath = [(row['fl'], 0, row['dur']) for _, row in subject_data.iterrows()]
        grouped_scanpaths.append(scanpath)
        subject_ids.append(subject_id)

    # Compute dissimilarity matrix
    dissimilarity_matrix = []
    for i, sp1 in enumerate(grouped_scanpaths):
        total_dissimilarity = 0
        count = 0
        for j, sp2 in enumerate(grouped_scanpaths):
            if i != j:  # Exclude self-comparison
                try:
                    dissimilarity = scasim(sp1, sp2)  # Calculate dissimilarity
                    total_dissimilarity += dissimilarity
                    count += 1
                except IndexError:
                    print(f"Issue with scanpath {i} or {j}. Skipping comparison.")

        if count > 0:
            average_dissimilarity = total_dissimilarity / count
            dissimilarity_matrix.append((i, average_dissimilarity))

    # Find the scanpath with the smallest average dissimilarity
    if dissimilarity_matrix:
        central_sp_index = min(dissimilarity_matrix, key=lambda x: x[1])[0]
        central_scanpath = grouped_scanpaths[central_sp_index]
        central_subject_id = subject_ids[central_sp_index]
        min_scasim_score = min(dissimilarity_matrix, key=lambda x: x[1])[1]
    else:
        central_scanpath = None  # Handle case where no valid scanpaths exist

    return central_scanpath, central_subject_id, min_scasim_score


def compute_scasim(dataset, location_preds, duration_preds, sn_id):
    most_central_scanpath, _, _ = _compute_central_sp(dataset, sn_id)
    # scasim()
    print("location_preds", location_preds)
    print("duration_preds", duration_preds)
    print("sn_id", sn_id)


def scasim(s, t, modulator=0.83):
    # modulator 0.83 (by default) - specifies how spatial distances between fixations
    # #' are assessed.  When set to 0, any spatial divergence of two
    # #' compared scanpaths is penalized independently of its degree.  When
    # #' set to 1, the scanpaths are compared only with respect to their
    # #' temporal patterns.  The default value approximates the sensitivity
    # #' to spatial distance found in the human visual system.
    # TODO: need to write a separate function that can take dfs as a parameter in order to compute the pair-wise dissimilarities in the dataset

    # Prepare matrix:
    m, n = len(s), len(t)
    d = [list(map(lambda i: 0, range(n + 1))) for _ in range(m + 1)]

    # Sequence alignment
    acc = 0
    # loop over fixations in scanpath s:
    for i in range(1,m+1):
        acc += s[i-1][2]
        d[i][0] = acc

    # loop over fixations in scanpath t:
    acc = 0
    for j in range(1,n+1):
        acc += t[j-1][2]
        d[0][j] = acc

    # Compute similarity:
    for i in range(n):
        for j in range(m):
            # calculating angle between fixation targets:
            slon = s[j][0] / (180/pi) # longitude (x-axis)
            tlon = t[i][0] / (180/pi)
            slat = s[j][1] / (180/pi) # latitude (y-axis)
            tlat = t[i][1] / (180/pi)

            angle = acos(sin(slat) * sin(tlat) + cos(slat) * cos(tlat) * cos(slon - tlon)) * (180/pi)

            # approximation of cortical magnification:
            mixer = modulator**angle

            # cost for substitution:
            cost = (abs(t[i][2] - s[j][2]) * mixer +
                       (t[i][2] + s[j][2]) * (1.0 - mixer))

            # select optimal edit operation
            ops = (d[j][i+1] + s[j][2],
                   d[j+1][i] + t[i][2],
                   d[j][i] + cost)

            mi = which_min(*ops)

            d[j+1][i+1] = ops[mi]

    return d[-1][-1]

def revert_z_score_normalization(data_list, mean, std):
    reverted_data = []
    for item in data_list:
        # Convert item to a PyTorch tensor
        item = torch.tensor(item)
        # Exclude the first and last values
        values_to_revert = item[1:-1]  # Exclude first and last value

        # Revert z-score normalization
        reverted_values = (values_to_revert * std) + mean
        reverted_values *= 1000
        reverted_values = torch.round(reverted_values).to(torch.int32)

        # Concatenate back with first and last values
        reverted_list = torch.cat([item[:1], reverted_values, item[-1:]], dim=0)
        reverted_data.append(reverted_list.tolist())

    return reverted_data


def prepare_scanpath(sp_dnn, dur_dnn, sn_len, sp_human, sp_fix_dur_test, cf, sn_ids_test, fix_dur_mean, fix_dur_std):
    max_sp_len = sp_dnn.shape[1]
    sp_human = sp_human.detach().to('cpu').numpy()
    # Find the number "sn_len+1" -> the end point
    stop_indx = []
    for i in range(sp_dnn.shape[0]):
        stop = np.where(sp_dnn[i,:]==(sn_len[i]+1))[0]
        if len(stop)== 0:  # no end point can be find -> exceeds the maximum length of the generated scanpath
            stop_indx.append(max_sp_len-1)
        else:
            stop_indx.append(stop[0])

    # Truncating data after the end point
    sp_dnn_cut = [sp_dnn[i][:stop_indx[i]+1] for i in range(sp_dnn.shape[0])]
    # replace the last terminal number to cf["max_sn_len"]-1, keep the same as the human scanpath label
    for i in range(len(sp_dnn_cut)):
        sp_dnn_cut[i][-1] = cf["max_sn_len"]-1  # the last location is always 26 (end token)
    # Truncate DNN duration data after the end point
    dur_dnn_cut = [dur_dnn[i][:stop_indx[i] + 1] for i in range(dur_dnn.shape[0])]
    print("generated durations", dur_dnn_cut)

    # process the human scanpath data, truncating data after the end point
    stop_indx = [np.where(sp_human[i,:]==cf["max_sn_len"]-1)[0][0] for i in range(sp_human.shape[0])]
    sp_human_cut = [sp_human[i][:stop_indx[i]+1] for i in range(sp_human.shape[0])]
    # Truncate human duration data after the end point
    sp_fix_dur_test_cut = [sp_fix_dur_test[i][:stop_indx[i] + 1] for i in range(sp_fix_dur_test.shape[0])]
    print("human durations", sp_fix_dur_test_cut)

    # Revert the z-score normalisation for durations
    # human durations
    sp_fix_dur_test_cut = revert_z_score_normalization(sp_fix_dur_test_cut, fix_dur_mean, fix_dur_std)
    print("human dur after z-score", sp_fix_dur_test_cut)
    # generated durations
    dur_dnn_cut = revert_z_score_normalization(dur_dnn_cut, fix_dur_mean, fix_dur_std)
    print("generated dur after z-score", dur_dnn_cut)

    # Create dictionaries to store the scanpath data
    sp_dnn_dict = {'locations': sp_dnn_cut, 'durations': dur_dnn_cut, 'sent_id': sn_ids_test.tolist()}
    sp_human_dict = {'locations': sp_human_cut, 'durations': sp_fix_dur_test_cut, 'sent_id': sn_ids_test.tolist()}

    # TODO: need to set duration to 0 when location is 26 (end token) for sp_dnn

    return sp_dnn_dict, sp_human_dict



def celer_load_native_speaker():
    sub_metadata_path = './Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    native_sub_list = sub_infor[sub_infor.L1 == 'English'].List.values
    return native_sub_list.tolist()

def compute_word_length_celer(arr):
    #length of a punctuation is 0, plus an epsilon to avoid division output inf
    arr = arr.astype('float64')
    arr[arr==0] = 1/(0+0.5)
    arr[arr!=0] = 1/(arr[arr!=0])
    return arr

def _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
    """
    SN_token_embedding   <CLS>, bla, bla, <SEP>
    SP_token_embedding       <CLS>, bla, bla, <SEP>
    SP_ordinal_pos 0, bla, bla, max_sp_len
    SP_fix_dur     0, bla, bla, 0
    """
    SN_ids = []
    SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
    SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
    SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
    sub_id_list =  []
    for sn_id in tqdm(sn_list):
        #process sentence sequence
        sn_df = eyemovement_df[eyemovement_df.sentenceid==sn_id]
        #notice: Each sentence is recorded multiple times in file |word_info_df|.
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        sn = sn[sn['list']==sn.list.values.tolist()[0]]
        #compute word length for each word
        sn_word_len = compute_word_length_celer(sn.WORD_LEN.values)

        sn_str = sn.sentence.iloc[-1]
        #nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
        if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
            #extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]
        sn_len = len(sn_str.split())

        #tokenization and padding
        tokenizer.padding_side = 'right'
        sn_str = '[CLS]' + ' ' + sn_str + ' ' + '[SEP]'
        #pre-tokenized input
        tokens = tokenizer.encode_plus(sn_str.split(),
                                        add_special_tokens = False,
                                        truncation=False,
                                        max_length = cf['max_sn_token'],
                                        padding = 'max_length',
                                        return_attention_mask=True,
                                        is_split_into_words=True)
        encoded_sn = tokens['input_ids']
        mask_sn = tokens['attention_mask']
        #use offset mapping to determine if two tokens are in the same word.
        #index start from 0, CLS -> 0 and SEP -> last index
        word_ids_sn = tokens.word_ids()
        word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

        #process fixation sequence
        for sub_id in reader_list:
            sub_df = sn_df[sn_df.list==sub_id]
            # remove fixations on non-words
            sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
            if len(sub_df) == 0:
                #no scanpath data found for the subject
                continue

            #prepare decoder input and output
            sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

            #dataset is noisy -> sanity check
            # 1) check if recorded fixation duration are within reasonable limits
            #Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
            outlier_indx = np.where(sp_fix_dur<50)[0]
            if outlier_indx.size>0:
                for out_idx in range(len(outlier_indx)):
                    outlier_i = outlier_indx[out_idx]
                    merge_flag = False

                    #outliers are commonly found in the fixation of the last record and the first record, and are removed directly
                    if outlier_i == len(sp_fix_dur)-1 or outlier_i == 0:
                        merge_flag = True

                    else:
                        if outlier_i-1 >= 0 and merge_flag == False:
                            #try to merge with the left fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i-1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i-1] = sp_fix_dur[outlier_i-1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i+1 < len(sp_fix_dur) and merge_flag == False:
                            #try to merge with the right fixation
                            if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i+1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i+1] = sp_fix_dur[outlier_i+1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx-1

            # 2) scanpath too long, remove outliers, speed up the inference
            if len(sp_word_pos) > 50: # 72/10684
                continue
            # 3)scanpath too short for a normal length sentence
            if len(sp_word_pos)<=1 and sn_len>10:
                continue

            # 4) check landing position feature
            #assign missing value to 'nan'
            sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc)
            #convert string of number of float type
            sp_fix_loc = [float(i) for i in sp_fix_loc]

            #Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
            if np.nanmax(sp_fix_loc)>35:
                missing_idx = np.where(np.array(sp_fix_loc)>5)[0]
                for miss in missing_idx:
                    if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                        sp_fix_loc[miss] = np.nan
                    else:
                        print('Landing position calculation error. Unknown cause, needs to be checked')

            sp_ordinal_pos = sp_word_pos.astype(int)
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_ordinal_pos.append(sp_ordinal_pos)
            SP_fix_dur.append(sp_fix_dur)
            SP_landing_pos.append(sp_fix_loc)

            sp_token = [sn_str.split()[int(i)] for i in sp_ordinal_pos]
            sp_token_str = '[CLS]' + ' ' + ' '.join(sp_token) + ' ' + '[SEP]'

            #tokenization and padding for scanpath, i.e. fixated word sequence
            sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
                                            add_special_tokens = False,
                                            truncation=False,
                                            max_length = cf['max_sp_token'],
                                            padding = 'max_length',
                                            return_attention_mask=True,
                                            is_split_into_words=True)
            encoded_sp = sp_tokens['input_ids']
            mask_sp = sp_tokens['attention_mask']
            #index start from 0, CLS -> 0 and SEP -> last index
            word_ids_sp = sp_tokens.word_ids()
            word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
            SP_input_ids.append(encoded_sp)
            SP_attention_mask.append(mask_sp)
            WORD_ids_sp.append(word_ids_sp)

            #sentence information
            SN_input_ids.append(encoded_sn)
            SN_attention_mask.append(mask_sn)
            SN_WORD_len.append(sn_word_len)
            WORD_ids_sn.append(word_ids_sn)
            sub_id_list.append(int(sub_id))
            SN_ids.append(sn_id)

    #padding for batch computation
    SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
    SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
    SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
    SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

    #assign type
    SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
    SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
    SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
    SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
    sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
    WORD_ids_sn = np.asarray(WORD_ids_sn)
    WORD_ids_sp = np.asarray(WORD_ids_sp)
    SN_ids = np.asarray(SN_ids)

    data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len, "WORD_ids_sn": WORD_ids_sn,
            "SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask, "WORD_ids_sp": WORD_ids_sp,
            "SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
            "sub_id": sub_id_list, "SN_ids": SN_ids
            }

    return data

class celerdataset(Dataset):
    """Return celer dataset."""

    def __init__(
        self,
        word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
    ):

        self.data = _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

    def __len__(self):
        return len(self.data["SN_input_ids"])


    def __getitem__(self,idx):
        sample = {}
        sample["sn_ids"] = self.data["SN_ids"][idx]
        sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
        sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
        sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
        sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

        sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
        sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
        sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

        sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
        sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
        sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

        sample["sub_id"] = self.data["sub_id"][idx]

        return sample


def one_hot_encode(arr, dim):
    # one hot encode
    onehot_encoded = np.zeros((arr.shape[0], dim))
    for idx, value in enumerate(arr):
        onehot_encoded[idx, value] = 1

    return onehot_encoded



def gradient_clipping(dnn_model, clip = 10):
    torch.nn.utils.clip_grad_norm_(dnn_model.parameters(),clip)
