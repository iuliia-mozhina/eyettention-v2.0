import pandas as pd
import numpy as np
from tqdm import tqdm
import ast


def prepare_corpus_file():
    word_info_df = pd.read_csv('../../Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # filter sentences that were read only by native speakers
    sub_metadata_path = '../../Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values
    sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

    all_sent = pd.DataFrame(columns=["SENT_LEN", "WORD_LEN", "FREQ_BLLIP", "SENT_NUM"])
    for sn_id in tqdm(sn_list):
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

        # convert the sentence name to the sentence number
        # read the sentence mapping dict
        with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)
        # encode sent ids numerically based on the provided dict from the CELER main training
        sn['sentenceid'] = sn['sentenceid'].map(sent_dict)

        # Create a DataFrame for the current sentence
        current_sentence_df = pd.DataFrame({
            "SENT_LEN": [sn_len] * len(sn["WORD_LEN"].values),
            "WORD_LEN": sn["WORD_LEN"].values.astype(int),
            "FREQ_BLLIP": sn["FREQ_BLLIP"].values,
            "SENT_NUM": sn["sentenceid"].values.astype(int)
        })
        # Drop rows with NaN values
        current_sentence_df.dropna(inplace=True)
        # Drop rows where WORD_LEN is 0
        current_sentence_df = current_sentence_df[current_sentence_df['WORD_LEN'] != 0]
        # Replace infinity FREQ by 0.0
        current_sentence_df['FREQ_BLLIP'].replace([np.inf, -np.inf], 0.013, inplace=True)
        current_sentence_df['FREQ_BLLIP'] = current_sentence_df['FREQ_BLLIP'].astype(float)

        # Update the sentence length after dropping some columns
        current_sentence_df['SENT_LEN'] = len(current_sentence_df)

        # Append the current sentence data to all_sent
        all_sent = all_sent.append(current_sentence_df, ignore_index=True)

    all_sent.to_csv('corpus_celer.dat', sep='\t', index=False, header=False)


def prepare_sequence_file():
    word_info_df = pd.read_csv('../../Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
    eyemovement_df = pd.read_csv('../../Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t', low_memory=False)
    eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace(
        '\t(.*)', '', regex=True)

    # retrieve native speakers
    sub_metadata_path = '../../Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values

    for reader in tqdm(reader_list):
        # process fix data for a particular reader
        reader_df = eyemovement_df[eyemovement_df.list == reader]  # reader df for all sentences
        reader_df = reader_df[
            ['sentenceid', 'CURRENT_FIX_INTEREST_AREA_ID', 'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
             'CURRENT_FIX_DURATION', 'PREVIOUS_SAC_DURATION', 'CURRENT_FIX_INTEREST_AREA_LABEL',
             'CURRENT_FIX_INTEREST_AREA_LEFT']]
        # Create an empty DataFrame to accumulate data for each reader
        reader_all_sent = pd.DataFrame(
            columns=["sentenceid", "CURRENT_FIX_INTEREST_AREA_ID", "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
                     "CURRENT_FIX_DURATION", "PREVIOUS_SAC_DURATION"])
        # iterate through sentences read by the current subject
        for sn_id in tqdm(reader_df['sentenceid'].unique().tolist()):
            sn_df = reader_df[reader_df.sentenceid == sn_id]  # sentence df

            # convert the sentence name to the sentence number
            # read the sentence mapping dict
            with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
                data_str = f.read()
                sent_dict = ast.literal_eval(data_str)
            # encode sent ids numerically based on the provided dict from the CELER main training
            sn_df['sentenceid'] = sn_df['sentenceid'].map(sent_dict)
            # remove fixations of not words
            sn_df = sn_df.loc[sn_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
            if len(sn_df) == 0:
                # no scanpath data found for the subject
                continue

            # Replace all occurrences of '.' with 0.0 in CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE
            sn_df['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] = sn_df[
                'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'].replace('.', 0.0).astype(float)

            sp_word_pos, sp_fix_loc, sp_fix_dur = sn_df.CURRENT_FIX_INTEREST_AREA_ID.values, sn_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sn_df.CURRENT_FIX_DURATION.values
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
                            if sn_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sn_df.iloc[
                                outlier_i - 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i - 1] = sp_fix_dur[outlier_i - 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                        if outlier_i + 1 < len(sp_fix_dur) and merge_flag == False:
                            # try to merge with the right fixation
                            if sn_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sn_df.iloc[
                                outlier_i + 1].CURRENT_FIX_INTEREST_AREA_LABEL:
                                sp_fix_dur[outlier_i + 1] = sp_fix_dur[outlier_i + 1] + sp_fix_dur[outlier_i]
                                merge_flag = True

                    sp_word_pos = np.delete(sp_word_pos, outlier_i)
                    sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
                    sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
                    sn_df.drop(sn_df.index[outlier_i], axis=0, inplace=True)
                    outlier_indx = outlier_indx - 1

            # 2) scanpath too long, remove outliers, speed up the inference
            if len(sp_word_pos) > 50:  # 72/10684
                continue

            # 3)scanpath too short
            if len(sp_word_pos) <= 1:
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
                    if sn_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
                        sp_fix_loc[miss] = np.nan
                    else:
                        print('Landing position calculation error. Unknown cause, needs to be checked')

            sn_df['PREVIOUS_SAC_DURATION'] = pd.to_numeric(sn_df['PREVIOUS_SAC_DURATION'], errors='coerce').fillna(
                0).astype(int)

            # Append the current sentence data to the reader's DataFrame
            reader_all_sent = reader_all_sent.append(sn_df[["sentenceid", "CURRENT_FIX_INTEREST_AREA_ID",
                                                            "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
                                                            "CURRENT_FIX_DURATION", "PREVIOUS_SAC_DURATION"]],
                                                     ignore_index=True)

            # Create placeholder column (SWIFT input format) to mark the fix sequence boundaries
            reader_all_sent['placeholder'] = 0
            reader_all_sent.loc[0, 'placeholder'] = 1  # First row always 1
            for i in range(1, len(reader_all_sent)):
                if reader_all_sent.loc[i, 'sentenceid'] != reader_all_sent.loc[i - 1, 'sentenceid']:
                    reader_all_sent.loc[i, 'placeholder'] = 1
                if i < len(reader_all_sent) - 1 and reader_all_sent.loc[i, 'sentenceid'] != reader_all_sent.loc[
                    i + 1, 'sentenceid']:
                    reader_all_sent.loc[i, 'placeholder'] = 2
            if len(reader_all_sent) > 1:
                reader_all_sent.loc[len(reader_all_sent) - 1, 'placeholder'] = 2  # Last row always 2

        reader_all_sent['sentenceid'] = reader_all_sent['sentenceid'] + 1

        # Save the reader's sequence data to a .dat file
        reader_all_sent.to_csv(f'fixseqin_{reader}.dat', sep='\t', index=False, header=False)
        print(f'Saved data for reader {reader} to fixseqin_{reader}.dat')


def post_process_swift():
    df = pd.read_csv('SWIFT/SIM/seq_generation_new.dat', delimiter='\t', header=None)
    df.rename(columns={0: 'subj_id', 1: 'SN', 2: 'loc', 3: 'land_pos', 4: 'dur'}, inplace=True)
    df = df[['subj_id', 'SN', 'loc', 'land_pos', 'dur']]
    df['SN'] = df['SN'] - 1
    df.to_csv('fixseqin_generation_cleaned.csv', sep='\t', index=False)

