from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer
from model import Eyettention_readerID
from sklearn.preprocessing import LabelEncoder
from collections import deque
import pickle
import argparse
from scasim import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Eyettention readerID model')
    parser.add_argument(
        '--test_mode',
        help='test mode: text',
        type=str,
        default='text'
    )
    parser.add_argument(
        '--atten_type',
        help='attention type',
        type=str,
        default='local-g'
    )
    parser.add_argument(
        '--save_data_folder',
        help='folder path for saving results',
        type=str,
        default='./results/BSC/'
    )
    parser.add_argument(
        '--gpu',
        help='gpu index',
        type=int,
        default=3
    )
    parser.add_argument(
        '--emb_size',
        help='readerID embedding size',
        type=int,
        default=16
    )
    args = parser.parse_args()
    gpu = args.gpu

    torch.set_default_tensor_type('torch.FloatTensor')
    availbl = torch.cuda.is_available()
    print(torch.cuda.is_available())
    if availbl:
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'
    torch.cuda.set_device(gpu)

    cf = {"model_pretrained": "bert-base-chinese",
          "lr": 1e-3,
          "max_grad_norm": 10,
          "n_epochs": 1000,
          "n_folds": 5,
          "dataset": 'BSC',
          "atten_type": args.atten_type,
          "subid_emb_size": args.emb_size,
          "batch_size": 256,
          "max_sn_len": 27,  # include start token and end token
          "max_sp_len": 40,  # include start token and end token
          "norm_type": "z-score",
          "earlystop_patience": 20,
          }

    # Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
    le = LabelEncoder()
    le.fit(np.append(np.arange(-cf["max_sn_len"] + 3, cf["max_sn_len"] - 1), cf["max_sn_len"] - 1))
    # le.classes_

    # load corpus
    word_info_df, pos_info_df, eyemovement_df = load_corpus(cf["dataset"])
    # Make list with sentence index
    sn_list = np.unique(eyemovement_df.sn.values).tolist()
    # Make list with reader index
    reader_list = np.unique(eyemovement_df.id.values).tolist()

    # Split training&test sets by text
    print('Start evaluating on new sentences.')
    split_list = sn_list

    n_folds = cf["n_folds"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_indx = 0
    sp_dnn_list = []
    sp_human_list = []
    for train_idx, test_idx in kf.split(split_list):
        loss_dict = {'val_loss': [], 'train_loss': [], 'test_ll': [], 'test_ll_SE': [], 'test_mse_dur': [],
                     'test_mse_dur_SE': [], 'test_mse_land_pos': [], 'test_mse_land_pos_SE': [],
                     'central_scasim_dnn': [], 'central_scasim_dnn_SE': [], 'central_scasim_human': [],
                     'central_scasim_human_SE': [], 'scasim_dnn': [], 'scasim_dnn_SE': [], 'scasim_human': [],
                     'scasim_human_SE': []}
        list_train = [split_list[i] for i in train_idx]
        list_test = [split_list[i] for i in test_idx]

        # create train validation split for training the models:
        kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        for train_index, val_index in kf_val.split(list_train):
            # we only evaluate a single fold
            break
        list_train_net = [list_train[i] for i in train_index]
        list_val_net = [list_train[i] for i in val_index]

        sn_list_train = list_train_net
        sn_list_val = list_val_net
        sn_list_test = list_test
        reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list

        # initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(cf['model_pretrained'])
        # Preparing batch data
        dataset_train = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_train, sn_list_train, tokenizer)
        train_dataloaderr = DataLoader(dataset_train, batch_size=cf["batch_size"], shuffle=True, drop_last=True)

        dataset_val = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_val, sn_list_val, tokenizer)
        val_dataloaderr = DataLoader(dataset_val, batch_size=cf["batch_size"], shuffle=False, drop_last=True)

        dataset_test = BSCdataset(word_info_df, eyemovement_df, cf, reader_list_test, sn_list_test, tokenizer)
        test_dataloaderr = DataLoader(dataset_test, batch_size=cf["batch_size"], shuffle=False, drop_last=False)

        # z-score normalization for gaze features
        fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur",
                                                       padding_value=0, scale=1000)
        landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos",
                                                               padding_value=0)
        sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

        # load model here
        dnn = Eyettention_readerID(cf)

        # training
        episode = 0
        optimizer = Adam(dnn.parameters(), lr=cf["lr"])
        dnn.train()
        dnn.to(device)
        av_score = deque(maxlen=100)
        av_location_score = deque(maxlen=100)
        av_duration_score = deque(maxlen=100)
        av_land_pos_score = deque(maxlen=100)
        old_score = 1e10
        save_ep_couter = 0
        print('Start training')
        for episode_i in range(episode, cf["n_epochs"] + 1):
            dnn.train()
            print('episode:', episode_i)
            counter = 0
            for batchh in train_dataloaderr:
                counter += 1
                batchh.keys()
                sn_input_ids = batchh["sn_input_ids"].to(device)
                sn_attention_mask = batchh["sn_attention_mask"].to(device)
                sp_input_ids = batchh["sp_input_ids"].to(device)
                sp_attention_mask = batchh["sp_attention_mask"].to(device)
                sp_pos = batchh["sp_pos"].to(device)
                sp_landing_pos = batchh["sp_landing_pos"].to(device)
                sp_fix_dur = (batchh["sp_fix_dur"] / 1000).to(device)
                sn_word_len = batchh["sn_word_len"].to(device)
                sub_id = batchh["sub_id"].to(device)

                # normalize gaze features
                mask = ~torch.eq(sp_fix_dur, 0)
                sp_fix_dur = (sp_fix_dur - fix_dur_mean) / fix_dur_std * mask
                sp_landing_pos = (sp_landing_pos - landing_pos_mean) / landing_pos_std * mask
                sp_fix_dur = torch.nan_to_num(sp_fix_dur)
                sp_landing_pos = torch.nan_to_num(sp_landing_pos)
                sn_word_len = (sn_word_len - sn_word_len_mean) / sn_word_len_std
                sn_word_len = torch.nan_to_num(sn_word_len)

                # zero old gradients
                optimizer.zero_grad()
                # predict output with DNN
                location_preds, duration_preds, landing_pos_preds, atten_weights = dnn(sn_emd=sn_input_ids,
                                                                                       sn_mask=sn_attention_mask,
                                                                                       sp_emd=sp_input_ids,
                                                                                       sp_pos=sp_pos,
                                                                                       word_ids_sn=None,
                                                                                       word_ids_sp=None,
                                                                                       sp_fix_dur=sp_fix_dur,
                                                                                       sp_landing_pos=sp_landing_pos,
                                                                                       sn_word_len=sn_word_len,
                                                                                       sub_id=sub_id)

                location_preds = location_preds.permute(0, 2, 1)  # [batch, dec_o_dim, step]

                # prepare label and mask for locations
                pad_mask, label = load_label(sp_pos, cf, le, device)
                loss = nn.CrossEntropyLoss(reduction="none")
                batch_location_error = torch.mean(torch.masked_select(loss(location_preds, label), ~pad_mask))

                # Compute loss for fixation durations
                duration_labels = sp_fix_dur[:,
                                  :39]  # Adjust duration_labels to match the sequence length of duration_preds
                duration_preds = duration_preds.squeeze(-1)  # Remove extra dimension (from [256, 39, 1] to [256, 39])
                dur_loss = nn.MSELoss(reduction="none")
                batch_duration_error = torch.mean(dur_loss(duration_preds, duration_labels))

                # Compute loss for landing position
                landing_pos_labels = sp_landing_pos[:,
                                     :39]  # Adjust duration_labels to match the sequence length of duration_preds
                landing_pos_preds = landing_pos_preds.squeeze(
                    -1)  # Remove extra dimension (from [256, 39, 1] to [256, 39])
                land_pos_loss = nn.MSELoss(reduction="none")
                batch_land_pos_error = torch.mean(land_pos_loss(landing_pos_preds, landing_pos_labels))

                # Combined loss for both location and duration
                batch_error = batch_location_error + batch_duration_error + batch_land_pos_error

                # backpropagate loss
                batch_error.backward()
                # clip gradients
                gradient_clipping(dnn, cf["max_grad_norm"])

                # learn
                optimizer.step()
                av_location_score.append(batch_location_error.to('cpu').detach().numpy())
                av_duration_score.append(batch_duration_error.to('cpu').detach().numpy())
                av_land_pos_score.append(batch_land_pos_error.to('cpu').detach().numpy())
                av_score.append(batch_error.to('cpu').detach().numpy())
                print('counter:', counter)
                print(
                    '\rSample {}\tLocation Loss: {:.10f}\tDuration Loss: {:.10f}\tLanding position Loss: {:.10f}'.format(
                        counter, np.mean(av_location_score), np.mean(av_duration_score), np.mean(av_land_pos_score)),
                    end=" ")
            loss_dict['train_loss'].append(np.mean(av_score))

            location_val_loss = []
            duration_val_loss = []
            land_pos_val_loss = []
            val_loss = []
            dnn.eval()
            for batchh in val_dataloaderr:
                with torch.no_grad():
                    sn_input_ids_val = batchh["sn_input_ids"].to(device)
                    sn_attention_mask_val = batchh["sn_attention_mask"].to(device)
                    sp_input_ids_val = batchh["sp_input_ids"].to(device)
                    sp_attention_mask_val = batchh["sp_attention_mask"].to(device)
                    sp_pos_val = batchh["sp_pos"].to(device)
                    sp_landing_pos_val = batchh["sp_landing_pos"].to(device)
                    sp_fix_dur_val = (batchh["sp_fix_dur"] / 1000).to(device)
                    sn_word_len_val = batchh["sn_word_len"].to(device)
                    sub_id_val = batchh["sub_id"].to(device)

                    # normalize gaze features
                    mask = ~torch.eq(sp_fix_dur_val, 0)
                    sp_fix_dur_val = (sp_fix_dur_val - fix_dur_mean) / fix_dur_std * mask
                    sp_landing_pos_val = (sp_landing_pos_val - landing_pos_mean) / landing_pos_std * mask
                    sp_fix_dur_val = torch.nan_to_num(sp_fix_dur_val)
                    sp_landing_pos_val = torch.nan_to_num(sp_landing_pos_val)
                    sn_word_len_val = (sn_word_len_val - sn_word_len_mean) / sn_word_len_std
                    sn_word_len_val = torch.nan_to_num(sn_word_len_val)

                    location_preds_val, duration_preds_val, landing_pos_preds_val, atten_weights_val = dnn(
                        sn_emd=sn_input_ids_val,
                        sn_mask=sn_attention_mask_val,
                        sp_emd=sp_input_ids_val,
                        sp_pos=sp_pos_val,
                        word_ids_sn=None,
                        word_ids_sp=None,
                        sp_fix_dur=sp_fix_dur_val,
                        sp_landing_pos=sp_landing_pos_val,
                        sn_word_len=sn_word_len_val,
                        sub_id=sub_id_val)

                    location_preds_val = location_preds_val.permute(0, 2, 1)  # [batch, dec_o_dim, step

                    # prepare label and mask for locations
                    loss = nn.CrossEntropyLoss(reduction="none")
                    pad_mask_val, label_val = load_label(sp_pos_val, cf, le, device)
                    location_error_val = torch.mean(
                        torch.masked_select(loss(location_preds_val, label_val), ~pad_mask_val))
                    location_val_loss.append(location_error_val.detach().to('cpu').numpy())

                    # Compute duration prediction error
                    duration_labels_val = sp_fix_dur_val[:,
                                          :39]  # Adjust duration_labels to match the sequence length of duration_preds
                    duration_preds_val = duration_preds_val.squeeze(-1)
                    duration_error_val = torch.mean(dur_loss(duration_preds_val, duration_labels_val))
                    duration_val_loss.append(duration_error_val.detach().to('cpu').numpy())

                    # Compute loss for landing position
                    landing_pos_labels_val = sp_landing_pos_val[:,
                                             :39]  # Adjust duration_labels to match the sequence length of duration_preds
                    landing_pos_preds_val = landing_pos_preds_val.squeeze(
                        -1)  # Remove extra dimension (from [256, 39, 1] to [256, 39])
                    land_pos_error_val = torch.mean(land_pos_loss(landing_pos_preds_val, landing_pos_labels_val))
                    land_pos_val_loss.append(land_pos_error_val.detach().to('cpu').numpy())

                    combined_loss = location_error_val + duration_error_val + land_pos_error_val
                    val_loss.append(combined_loss.detach().to('cpu').numpy())

            print('\nValidation loss for locations {} \n'.format(np.mean(location_val_loss)))
            print('\nValidation loss for duration {} \n'.format(np.mean(duration_val_loss)))
            print('\nValidation loss for landing position {} \n'.format(np.mean(land_pos_val_loss)))
            loss_dict['val_loss'].append(np.mean(val_loss))

            if np.mean(val_loss) < old_score:
                # save model if val loss is smallest
                torch.save(dnn.state_dict(),
                           '{}/BSC_text_eyettention_readerID_emb{}_fold{}.pth'.format(
                               args.save_data_folder, args.emb_size, fold_indx))
                old_score = np.mean(val_loss)
                print('\nsaved model state dict\n')
                save_ep_couter = episode_i
            else:
                # early stopping
                if episode_i - save_ep_couter >= cf["earlystop_patience"]:
                    break

        # evaluation
        dnn.eval()
        res_llh = []
        res_mse_dur = []
        res_mse_land_pos = []
        res_central_scasim_human = []
        res_central_scasim_dnn = []
        res_scasim_human = []
        res_scasim_dnn = []
        dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,
                                                    f'BSC_text_eyettention_readerID_emb{args.emb_size}_fold{fold_indx}.pth'),
                                       map_location='cpu'))
        dnn.to(device)
        batch_indx = 0
        for batchh in test_dataloaderr:
            with torch.no_grad():
                sn_ids_test = batchh["sn_ids"].to(device)
                sn_input_ids_test = batchh["sn_input_ids"].to(device)
                sn_attention_mask_test = batchh["sn_attention_mask"].to(device)
                sp_input_ids_test = batchh["sp_input_ids"].to(device)
                sp_attention_mask_test = batchh["sp_attention_mask"].to(device)
                sp_pos_test = batchh["sp_pos"].to(device)  # 28: '<Sep>', 29: '<'Pad'>'
                sp_landing_pos_test = batchh["sp_landing_pos"].to(device)
                sp_fix_dur_test = (batchh["sp_fix_dur"] / 1000).to(device)
                sn_word_len_test = batchh["sn_word_len"].to(device)
                sub_id_test = batchh["sub_id"].to(device)

                # normalize gaze features
                mask = ~torch.eq(sp_fix_dur_test, 0)
                sp_fix_dur_test = (sp_fix_dur_test - fix_dur_mean) / fix_dur_std * mask
                sp_landing_pos_test = (sp_landing_pos_test - landing_pos_mean) / landing_pos_std * mask
                sp_fix_dur_test = torch.nan_to_num(sp_fix_dur_test)
                sp_landing_pos_test = torch.nan_to_num(sp_landing_pos_test)
                sn_word_len_test = (sn_word_len_test - sn_word_len_mean) / sn_word_len_std
                sn_word_len_test = torch.nan_to_num(sn_word_len_test)

                location_preds_test, duration_preds_test, landing_pos_preds_test, atten_weights_test = dnn(
                    sn_emd=sn_input_ids_test,
                    sn_mask=sn_attention_mask_test,
                    sp_emd=sp_input_ids_test,
                    sp_pos=sp_pos_test,
                    word_ids_sn=None,
                    word_ids_sp=None,
                    sp_fix_dur=sp_fix_dur_test,
                    sp_landing_pos=sp_landing_pos_test,
                    sn_word_len=sn_word_len_test,
                    sub_id=sub_id_test)

                # We do not use nn.CrossEntropyLoss here to calculate the likelihood because it combines nn.LogSoftmax and nn.NLL,
                # while nn.LogSoftmax returns a log value based on e, we want 2 instead
                # m = nn.LogSoftmax(dim=2) -- base e, we want base 2
                m = nn.Softmax(dim=2)
                location_preds_test = m(location_preds_test).detach().to('cpu').numpy()

                # prepare label and mask
                pad_mask_test, label_test = load_label(sp_pos_test, cf, le, 'cpu')
                pred = location_preds_test.argmax(axis=2)
                # compute log likelihood for the batch samples
                res_batch = eval_log_llh(location_preds_test, label_test, pad_mask_test)
                res_llh.append(np.array(res_batch))

                duration_labels_test = sp_fix_dur_test[:, :39]
                test_mask = mask[:, :39]
                duration_preds_test = duration_preds_test.squeeze(-1)
                mse_dur = eval_mse(duration_preds_test, duration_labels_test, test_mask)
                print("Mean MSE for durations", np.mean(mse_dur))
                res_mse_dur.append(np.array(mse_dur))

                landing_pos_preds_test = landing_pos_preds_test.squeeze(-1)
                landing_pos_labels_test = sp_landing_pos_test[:, :39]
                mse_landing_pos = eval_mse(landing_pos_preds_test, landing_pos_labels_test, test_mask)
                print("Mean MSE for landing positions", np.mean(mse_landing_pos))
                res_mse_land_pos.append(np.array(mse_landing_pos))

                if bool(args.scanpath_gen_flag) == True:
                    sn_len = (torch.sum(sn_attention_mask_test, axis=1) - 2).detach().to('cpu').numpy()
                    # compute the scan path generated from the model when the first CLS token is given
                    sp_dnn, _, dur_dnn, land_pos_dnn = dnn.scanpath_generation(sn_emd=sn_input_ids_test,
                                                                               sn_mask=sn_attention_mask_test,
                                                                               word_ids_sn=None,
                                                                               sn_word_len=sn_word_len_test,
                                                                               le=le,
                                                                               sp_fix_dur=sp_fix_dur_test,
                                                                               sp_landing_pos=sp_landing_pos_test,
                                                                               sub_id=sub_id_test,
                                                                               max_pred_len=cf['max_pred_len'])

                    sp_dnn, sp_human = prepare_scanpath(sp_dnn.detach().to('cpu').numpy(),
                                                        dur_dnn.detach().to('cpu').numpy(),
                                                        land_pos_dnn.detach().to('cpu').numpy(),
                                                        sn_len, sp_pos_test,
                                                        sp_fix_dur_test, sp_landing_pos_test, cf, sn_ids_test,
                                                        fix_dur_mean, fix_dur_std, landing_pos_mean, landing_pos_std)

                    sp_dnn_list.extend(sp_dnn)
                    sp_human_list.extend(sp_human)

                    sp_dnn = convert_sp_to_lists(sp_dnn)
                    sp_human = convert_sp_to_lists(sp_human)
                    sp_human = modify_landing_pos(sp_human.copy())
                    sp_dnn = modify_landing_pos(sp_dnn.copy())
                    random_sp = sample_random_sp("BSC", sp_human)
                    random_sp = convert_sp_to_lists(random_sp)
                    random_sp = modify_landing_pos(random_sp.copy())

                    scasim_scores_dnn = compute_scasim(sp_dnn, sp_human)
                    res_scasim_dnn.append(scasim_scores_dnn)
                    print("Mean scasim dnn", np.mean(scasim_scores_dnn))
                    scasim_scores_human = compute_scasim(sp_human, random_sp)
                    res_scasim_human.append(scasim_scores_human)
                    print("Mean scasim human", np.mean(scasim_scores_human))

                    central_scasim_scores_dnn = compute_central_scasim("BSC_most_central_sp.txt", sp_dnn)
                    central_scasim_scores_human = compute_central_scasim("BSC_most_central_sp.txt",
                                                                         sp_human)
                    res_central_scasim_dnn.append(np.array(central_scasim_scores_dnn))
                    res_central_scasim_human.append(np.array(central_scasim_scores_human))
                    print("Mean central scasim dnn", np.mean(central_scasim_scores_dnn))
                    print("Mean central scasim human", np.mean(central_scasim_scores_human))

                batch_indx += 1

        res_llh = np.concatenate(res_llh).ravel()
        loss_dict['test_ll'].append(res_llh)
        res_mse_dur = np.concatenate(res_mse_dur).ravel()
        loss_dict['test_mse_dur'].append(res_mse_dur)
        res_mse_land_pos = np.concatenate(res_mse_land_pos).ravel()
        loss_dict['test_mse_land_pos'].append(res_mse_land_pos)

        res_central_scasim_dnn = np.concatenate(res_central_scasim_dnn).ravel()
        loss_dict['central_scasim_dnn'].append(res_central_scasim_dnn)
        res_central_scasim_human = np.concatenate(res_central_scasim_human).ravel()
        loss_dict['central_scasim_human'].append(res_central_scasim_human)
        res_scasim_dnn = np.concatenate(res_scasim_dnn).ravel()
        loss_dict['scasim_dnn'].append(res_scasim_dnn)
        res_scasim_human = np.concatenate(res_scasim_human).ravel()
        loss_dict['scasim_human'].append(res_scasim_human)

        loss_dict['fix_dur_mean'] = fix_dur_mean
        loss_dict['fix_dur_std'] = fix_dur_std
        loss_dict['landing_pos_mean'] = landing_pos_mean
        loss_dict['landing_pos_std'] = landing_pos_std
        loss_dict['sn_word_len_mean'] = sn_word_len_mean
        loss_dict['sn_word_len_std'] = sn_word_len_std

        print('Test likelihood is {}'.format(np.mean(res_llh)))
        loss_dict['test_ll_SE'].append(np.std(res_llh) / np.sqrt(len(res_llh)))
        print("Standard error for NLL", np.std(res_llh) / np.sqrt(len(res_llh)))

        print('Test MSE for durations is {}'.format(np.mean(res_mse_dur)))
        loss_dict['test_mse_dur_SE'].append(np.std(res_mse_dur) / np.sqrt(len(res_mse_dur)))
        print("Standard error for MSE dur", np.std(res_mse_dur) / np.sqrt(len(res_mse_dur)))

        print('Test MSE for landing positions is {}'.format(np.mean(res_mse_land_pos)))
        loss_dict['test_mse_land_pos_SE'].append(np.std(res_mse_land_pos) / np.sqrt(len(res_mse_land_pos)))
        print("Standard error for MSE land pos", np.std(res_mse_land_pos) / np.sqrt(len(res_mse_land_pos)))

        print("Central Scasim dnn", np.mean(loss_dict['central_scasim_dnn']))
        loss_dict['central_scasim_dnn_SE'].append(np.std(res_central_scasim_dnn) / np.sqrt(len(res_central_scasim_dnn)))
        print("Standard error for Central scasim DNN",
              np.std(res_central_scasim_dnn) / np.sqrt(len(res_central_scasim_dnn)))

        print("Central Scasim human", np.mean(loss_dict['central_scasim_human']))
        loss_dict['central_scasim_human_SE'].append(
            np.std(res_central_scasim_human) / np.sqrt(len(res_central_scasim_human)))
        print("Standard error for Central scasim human",
              np.std(res_central_scasim_human) / np.sqrt(len(res_central_scasim_human)))

        print("Scasim dnn", np.mean(loss_dict['scasim_dnn']))
        loss_dict['scasim_dnn_SE'].append(np.std(res_scasim_dnn) / np.sqrt(len(res_scasim_dnn)))
        print("Standard error for scasim dnn", np.std(res_scasim_dnn) / np.sqrt(len(res_scasim_dnn)))

        print("Scasim human", np.mean(loss_dict['scasim_human']))
        loss_dict['scasim_human_SE'].append(np.std(res_scasim_human) / np.sqrt(len(res_scasim_human)))
        print("Standard error for scasim human", np.std(res_scasim_human) / np.sqrt(len(res_scasim_human)))

        # save results
        with open('{}/res_BSC_NRS_eyettention_readerID_emb{}_Fold{}.pickle'.format(args.save_data_folder,
                                                                                   args.emb_size,
                                                                                   fold_indx), 'wb') as handle:
            pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fold_indx += 1
