import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from transformers import BertModel, BertConfig


class Eyettention(nn.Module):
    def __init__(self, cf):
        super(Eyettention, self).__init__()
        self.cf = cf  # config dict
        self.window_width = 1  # D
        self.atten_type = cf["atten_type"]
        self.hidden_size = 128

        #############   Word-Sequence Encoder   ################
        encoder_config = BertConfig.from_pretrained(
            self.cf["model_pretrained"])  # you can specify the pre-trained model in the config
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print("keeping Bert with pre-trained weights")
        self.encoder = BertModel.from_pretrained(self.cf["model_pretrained"], config=encoder_config)
        self.encoder.eval()
        # freeze the parameters in Bert model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(0.4)
        self.encoder_lstm = nn.LSTM(input_size=768,  # BERT embedding size
									hidden_size=int(self.hidden_size / 2),
									num_layers=8,  # 8 Bi-LSTM layers (see model arch)
									batch_first=True,
									bidirectional=True,
									dropout=0.2)

        ################## Fixation-Sequence Encoder #################
        self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)

        # The scanpath is generated in an autoregressive manner, the output of the previous timestep is fed to the input of the next time step.
        # So we use decoder cells and loop over all timesteps.
        # initialize eight Location decoder cells
        self.decoder_cell1 = nn.LSTMCell(768 + 2,
                                         self.hidden_size)  # first layer input size = #BERT embedding size + two fixation attributes:landing position and fixiation duration
        self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.dropout_LSTM = nn.Dropout(0.2)

        # Cross-Attention
        # computes attention weights between the current hidden state of the fixation-sequence encoder (ht)
        # and the output from the word-sequence encoder (hs).
        self.attn = nn.Linear(self.hidden_size, self.hidden_size + 3)  # +3 account for the word length, word freq and predictability features

        # Decoder
        # Location prediction head
        self.dropout_dense = nn.Dropout(0.2)
        self.decoder_dense = nn.Sequential(
            self.dropout_dense,
            nn.Linear(self.hidden_size * 2 + 3, 512),  # was only + 1
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cf["max_sn_len"] * 2 - 3)
        )

        ################## Duration Decoder Cells #################
        self.decoder_duration_cell1 = nn.LSTMCell(768 + 51, self.hidden_size)
        self.decoder_duration_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_duration_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.dropout_LSTM_duration = nn.Dropout(0.2)

        # Duration prediction head
        self.dropout_duration = nn.Dropout(0.2)
        self.decoder_duration = nn.Sequential(
           self.dropout_duration,
            nn.Linear(self.hidden_size * 2 + 3, 512),
            nn.ReLU(),
            self.dropout_duration,
            nn.Linear(512, 128),
            nn.ReLU(),
            self.dropout_duration,
            nn.Linear(128, 128),
            nn.ReLU(),
            self.dropout_duration,
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output single value for fixation duration
        )

        # for scanpath generation
        self.softmax = nn.Softmax(dim=1)

        self.layer_norm_word = nn.LayerNorm(normalized_shape=131)

     #   self.layer_norm_dur = nn.LayerNorm(normalized_shape=819, eps=encoder_config.layer_norm_eps)

    def pool_subword_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
        # batching computing
        # Pool bert token (subword) to word level
        if target == 'sn':
            max_len = self.cf["max_sn_len"]  # CLS and SEP included
        elif target == 'sp':
            max_len = self.cf["max_sp_len"] - 1  # do not account the 'SEP' token

        merged_word_emb = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
        for word_idx in range(max_len):
            word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
            # pooling method -> sum
            if pool_method == 'sum':
                pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            elif pool_method == 'mean':
                pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

        mask_word = torch.sum(merged_word_emb, 2).bool()
        return merged_word_emb, mask_word

    def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len, sn_word_freq, sn_pred):
        # Word-Sequence Encoder
        outputs = self.encoder(input_ids=sn_emd, attention_mask=sn_mask)
        hidden_rep_orig, pooled_rep = outputs[0], outputs[1]
        if word_ids_sn != None:
            # Pool bert subword to word level for english corpus
            merged_word_emb, sn_mask_word = self.pool_subword_to_word(hidden_rep_orig,
                                                                      word_ids_sn,
                                                                      target='sn',
                                                                      pool_method='sum')
        else:  # no pooling for Chinese corpus
            merged_word_emb, sn_mask_word = hidden_rep_orig, None

        hidden_rep = self.embedding_dropout(merged_word_emb)
        x, (hn, hc) = self.encoder_lstm(hidden_rep, None)  # [256, 27, 128]

        # concatenate with the word length feature
        x = torch.cat((x, sn_word_len[:, :, None]), dim=2)  # [256, 27, 129]

        # concatenate with the word frequency feature
        x = torch.cat((x, sn_word_freq[:, :, None]), dim=2)  # [256, 27, 130]

        # concatenate with the predictability feature
        x = torch.cat((x, sn_pred[:, :, None]), dim=2)  # [256, 27, 131]

        # Apply LayerNorm along the last dimension (131)
        x = self.layer_norm_word(x)

        return x, sn_mask_word

    def cross_attention(self, ht, hs, sn_mask, cur_word_index):
        # General Attention:
        # score(ht,hs) = (ht^T)(Wa)hs
        # hs is the output from word-Sequence Encoder
        # ht is the previous hidden state from Fixation-Sequence Encoder
        # self.attn(o): [batch, step, units]
        attn_prod = torch.matmul(self.attn(ht.unsqueeze(1)), hs.permute(0, 2, 1))  # [batch, 1, step]
        if self.atten_type == 'global':  # global attention
            attn_prod += (~sn_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

        else:  # local attention
            # current fixated word index
            aligned_position = cur_word_index
            # Get window borders
            left = torch.where(aligned_position - self.window_width >= 0, aligned_position - self.window_width, 0)
            right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"] - 1,
                                aligned_position + self.window_width, self.cf["max_sn_len"] - 1)

            # exclude padding tokens
            # only consider words in the window
            sen_seq = torch.arange(self.cf["max_sn_len"])[None, :].expand(sn_mask.shape[0], self.cf["max_sn_len"]).to(
                sn_mask.device)
            outside_win_mask = (sen_seq < left.unsqueeze(1)) + (sen_seq > right.unsqueeze(1))

            attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

            if self.atten_type == 'local-g':  # local attention with Gaussian Kernel
                gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (
                            2 * torch.square(torch.tensor(self.window_width / 2))))
                gauss_factor = gauss(sen_seq)
                att_weight = att_weight * gauss_factor.unsqueeze(1)

        return att_weight

    def decode_location(self, sp_emd, sn_mask, sp_pos, enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp):
        ############ Decoder for fixation location #############
        # processes the fixation embedding input (sp_emd) along with gaze features (sp_fix_dur, sp_landing_pos)
        hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hx, cx = hn[0, :, :], hc[0, :, :]
        hx2, cx2 = hn[1, :, :], hc[1, :, :]
        hx3, cx3 = hn[2, :, :], hc[2, :, :]
        hx4, cx4 = hn[3, :, :], hc[3, :, :]
        hx5, cx5 = hn[4, :, :], hc[4, :, :]
        hx6, cx6 = hn[5, :, :], hc[5, :, :]
        hx7, cx7 = hn[6, :, :], hc[6, :, :]
        hx8, cx8 = hn[7, :, :], hc[7, :, :]

        # dec_emb_in represents word embeddings
        dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])
        if word_ids_sp is not None:
            # Pool bert subword to word level for English corpus
            sp_merged_word_emd, sp_mask_word = self.pool_subword_to_word(dec_emb_in,
                                                                         word_ids_sp[:, :-1],
                                                                         target='sp',
                                                                         pool_method='sum')
        else:  # no pooling for Chinese corpus
            sp_merged_word_emd, sp_mask_word = dec_emb_in, None

        # add positional embeddings
        position_embeddings = self.position_embeddings(sp_pos[:, :-1])
        dec_emb_in = sp_merged_word_emd + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)
        dec_emb_in = dec_emb_in.permute(1, 0, 2)  # [step, n, emb_dim]
        dec_emb_in = self.embedding_dropout(dec_emb_in)
        # at this stage dec_emb_in is word embeddings with positional embeddings

        # concatenate two additional gaze features
        if sp_landing_pos is not None:
            dec_emb_in = torch.cat((dec_emb_in, sp_landing_pos.permute(1, 0)[:-1, :, None]), dim=2)

        if sp_fix_dur is not None:
            dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1, 0)[:-1, :, None]), dim=2)

        # dec_emb_in is now word embeddings + duration + landing pos

        # Predict location output for each time step in turn
        location_output = []
        # save attention scores for visualization
        atten_weights_batch = torch.empty(sp_emd.shape[0], 0, self.cf["max_sn_len"]).to(sp_emd.device)

       # print("dec_emb_in", dec_emb_in.shape) # [39, 256, 770]

        for i in range(dec_emb_in.shape[0]):
            hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))  # [batch, units]
            hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
            hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
            hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(hx3), (hx4, cx4))
            hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(hx4), (hx5, cx5))
            hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(hx5), (hx6, cx6))
            hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(hx6), (hx7, cx7))
            hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(hx7), (hx8, cx8))

            # computes attention weights between the current hidden state of the fixation-sequence encoder (ht)
            # and the output from the word-sequence encoder (hs).
            att_weight = self.cross_attention(ht=hx8,  # current hidden state of the fixation-sequence encoder
                                              hs=enc_out, # output from the word-sequence encoder
                                              sn_mask=sn_mask,
                                              cur_word_index=sp_pos[:, i])  # [256, 1, 27]
            atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)  # [256, i, 27]

            context = torch.matmul(att_weight, enc_out)  # [batch, 1, units]

            # Decoder
            hc = torch.cat([context.squeeze(1), hx8], dim=1)  # [batch, units *2]
            location_result = self.decoder_dense(hc)  # [batch, dec_o_dim] [256, 51]
            location_output.append(location_result)

        location_output = torch.stack(location_output, dim=0)  # [step, batch, dec_o_dim]

        return location_output.permute(1, 0, 2), atten_weights_batch

    def decode_duration(self, dec_emb_in, location_output, enc_out_dur, sn_mask, sp_pos, sp_emd):
        ############ Decoder for fixation duration #############
        # Prepare input for duration prediction
        location_output = location_output.permute(1, 0, 2)  # [39, 256, 51]

        dur_emb_in = torch.cat([dec_emb_in, location_output], dim=2)  # [39, 256, 819]

     #   dur_emb_in = self.layer_norm_dur(dur_emb_in)  # layer normalisation

    #    print("dur_emb_in", dur_emb_in, dur_emb_in.shape)

        # Initialize hidden states for duration prediction LSTM cells
        batch_size = dur_emb_in.shape[1]  # 256
        hx_dur = torch.zeros(batch_size, self.hidden_size).to(dur_emb_in.device)
        cx_dur = torch.zeros(batch_size, self.hidden_size).to(dur_emb_in.device)

        # Predict duration output for each time step in turn
        duration_output = []

        # save attention scores for visualization
        atten_weights_batch_dur = torch.empty(sp_emd.shape[0], 0, self.cf["max_sn_len"]).to(sp_emd.device)

        for j in range(dur_emb_in.shape[0]):
            hx_dur, cx_dur = self.decoder_duration_cell1(dur_emb_in[j], (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell2(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell3(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell4(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell5(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell6(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell7(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell8(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))

            # before process only sp embeddings: location_output and dec_emb_in

            # computes attention weights between the current hidden state of the fixation-sequence encoder (ht)
            # and the output from the word-sequence encoder (hs).
            att_weight_dur = self.cross_attention(ht=hx_dur,  # current hidden state of the fixation-sequence encoder
                                              hs=enc_out_dur,  # output from the word-sequence encoder
                                              sn_mask=sn_mask,
                                              cur_word_index=sp_pos[:, j]) # [256, 1, 27]
            atten_weights_batch_dur = torch.cat([atten_weights_batch_dur, att_weight_dur], dim=1) # torch.Size([256, j, 27]

            context_dur = torch.matmul(att_weight_dur, enc_out_dur)  # [256, 1, 129] [batch, 1, units]
            # Decoder
            hc_dur = torch.cat([context_dur.squeeze(1), hx_dur], dim=1)  # [256, 257] correct
            duration_result = self.decoder_duration(hc_dur)  # location is [256, 51]
            duration_output.append(duration_result)

        duration_output = torch.stack(duration_output, dim=0)

        return duration_output.permute(1, 0, 2)

    def decode(self, sp_emd, sn_mask, sp_pos, enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp):
        # Phase 1: Decode fixation location
        location_output, atten_weights_batch = self.decode_location(sp_emd, sn_mask, sp_pos, enc_out, sp_fix_dur,
                                                                    sp_landing_pos,
                                                                    word_ids_sp)

        # Phase 2: Decode fixation duration using learned representations from location prediction
        dec_emb_in = self.prepare_input_for_duration_prediction(sp_emd, sp_pos, word_ids_sp)

        duration_output = self.decode_duration(dec_emb_in, location_output, enc_out, sn_mask, sp_pos, sp_emd)

        return location_output, duration_output, atten_weights_batch

    def prepare_input_for_duration_prediction(self, sp_emd, sp_pos, word_ids_sp):
        # dec_emb_in represents word embeddings with positional embeddings
        dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])
        if word_ids_sp is not None:
            # Pool bert subword to word level for English corpus
            sp_merged_word_emd, sp_mask_word = self.pool_subword_to_word(dec_emb_in,
                                                                         word_ids_sp[:, :-1],
                                                                         target='sp',
                                                                         pool_method='sum')
        else:  # no pooling for Chinese corpus
            sp_merged_word_emd, sp_mask_word = dec_emb_in, None

        # Add positional embeddings
        position_embeddings = self.position_embeddings(sp_pos[:, :-1])
        dec_emb_in = sp_merged_word_emd + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)
        dec_emb_in = dec_emb_in.permute(1, 0, 2)  # [step, n, emb_dim]
        dec_emb_in = self.embedding_dropout(dec_emb_in)

        return dec_emb_in

    def forward(self, sn_emd, sn_mask, sp_emd, sp_pos, word_ids_sn, word_ids_sp, sp_fix_dur, sp_landing_pos,
                sn_word_len, sn_word_freq, sn_pred):
        # Word-sequence encoder
        x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len, sn_word_freq, sn_pred)  # [batch, step, units], [batch, units]

        if sn_mask_word is None:  # for Chinese dataset without token pooling
            sn_mask = torch.Tensor.bool(sn_mask)
            location_preds, duration_preds, atten_weights = self.decode(sp_emd,
                                                                        sn_mask,
                                                                        sp_pos,
                                                                        x,
                                                                        sp_fix_dur,
                                                                        sp_landing_pos,
                                                                        word_ids_sp
                                                                        )  # [batch, step, dec_o_dim]

        else:  # for English dataset with token pooling
            location_preds, duration_preds, atten_weights = self.decode(sp_emd,
                                                                        sn_mask_word,
                                                                        sp_pos,
                                                                        x,
                                                                        sp_fix_dur,
                                                                        sp_landing_pos,
                                                                        word_ids_sp)  # [batch, step, dec_o_dim]

        return location_preds, duration_preds, atten_weights

    def scanpath_generation(self, sn_emd,
                            sn_mask,
                            word_ids_sn,
                            sn_word_len,
                            le,
                            sn_word_freq,
                            sn_pred,
                            max_pred_len=60
                            ):
        ############# Fixation location generation ##############
        # compute the scan path generated from the model when the first CLS taken is given
        enc_out, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len, sn_word_freq, sn_pred)
        if sn_mask_word is None:
            sn_mask = torch.Tensor.bool(sn_mask)
        else:
            sn_mask = sn_mask_word
        sn_len = torch.sum(sn_mask, axis=1) - 2

        # decode fixation locations
        # Initialize hidden state and cell state with zeros,
        hn = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
        hc = torch.zeros(8, sn_emd.shape[0], self.hidden_size).to(sn_emd.device)
        hx, cx = hn[0, :, :], hc[0, :, :]
        hx2, cx2 = hn[1, :, :], hc[1, :, :]
        hx3, cx3 = hn[2, :, :], hc[2, :, :]
        hx4, cx4 = hn[3, :, :], hc[3, :, :]
        hx5, cx5 = hn[4, :, :], hc[4, :, :]
        hx6, cx6 = hn[5, :, :], hc[5, :, :]
        hx7, cx7 = hn[6, :, :], hc[6, :, :]
        hx8, cx8 = hn[7, :, :], hc[7, :, :]

        batch_size = sn_emd.shape[0]  # 256
        hx_dur = torch.zeros(batch_size, self.hidden_size).to(sn_emd.device)
        cx_dur = torch.zeros(batch_size, self.hidden_size).to(sn_emd.device)

        # use CLS token (101) as start token
        dec_in_start = (torch.ones(sn_mask.shape[0]) * 101).long().to(sn_mask.device)
        dec_emb_in = self.encoder.embeddings.word_embeddings(dec_in_start)  # [batch, emb_dim]

        # add positional embeddings
        start_pos = torch.zeros(sn_mask.shape[0]).to(sn_mask.device)
        position_embeddings = self.position_embeddings(start_pos.long())
        dec_emb_in = dec_emb_in + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)

        # concatenate two additional gaze features, which are set to zeros for CLS token
        dec_in = torch.cat((dec_emb_in, torch.zeros(dec_emb_in.shape[0], 2).to(sn_emd.device)), dim=1)

        # generate fixation one by one in an autoregressive way
        output = []
        density_prediction = []
        location_output = []
        duration_output = []
        dec_emb_in_accumulated = []
        pred_counter = 0
        output.append(start_pos.long())  # append 0 at position 0 (len = 256)
        for p in range(max_pred_len - 1):
            dec_emb_in_accumulated.append(dec_emb_in)
            hx, cx = self.decoder_cell1(dec_in, (hx, cx))  # [batch, units]
            hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
            hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
            hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(hx3), (hx4, cx4))
            hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(hx4), (hx5, cx5))
            hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(hx5), (hx6, cx6))
            hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(hx6), (hx7, cx7))
            hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(hx7), (hx8, cx8))

            att_weight = self.cross_attention(ht=hx8,
                                              hs=enc_out,
                                              sn_mask=sn_mask,
                                              cur_word_index=output[-1])

            context = torch.matmul(att_weight, enc_out)  # [batch, 1, units]
            hc = torch.cat([context.squeeze(1), hx8], dim=1)  # [batch, units *2]

            result = self.decoder_dense(hc)  # [batch, dec_o_dim] torch.Size([256, 51])
            location_output.append(result)

            softmax_result = self.softmax(result)  # [batch, dec_o_dim]
            density_prediction.append(softmax_result)

            # we can either take argmax or sampling from the output distribution,
            # we do sampling in the paper
            # pred_indx = result.argmax(dim=1)
            # sampling next fixation location according to the distribution
            pred_indx = torch.multinomial(softmax_result, 1)
            pred_class = [le.classes_[pred_indx[i]] for i in torch.arange(softmax_result.shape[0])]
            pred_class = torch.from_numpy(np.array(pred_class)).to(sn_emd.device)
            # predict fixation word index = last fixation word index + predicted saccade range
            pred_pos = output[-1] + pred_class

            # larger than sentence max length -- set to sentence length+1, i.e. token <'SEP'>
            # prepare the input to the next timestep
            input_ids = []
            for i in range(pred_pos.shape[0]):
                if pred_pos[i] > sn_len[i]:
                    pred_pos[i] = sn_len[i] + 1
                elif pred_pos[i] < 1:
                    pred_pos[i] = 1

                if word_ids_sn is not None:
                    input_ids.append(sn_emd[i, word_ids_sn[i, :] == pred_pos[i]])
                else:
                    input_ids.append(sn_emd[i, pred_pos[i]])
            output.append(pred_pos)

            # prepare next timestamp input token
            pred_counter += 1
            if word_ids_sn is not None:
                # merge tokens
                dec_emb_in = torch.empty(0, 768).to(sn_emd.device)
                for id in input_ids:
                    dec_emb_in = torch.cat(
                        [dec_emb_in, torch.sum(self.encoder.embeddings.word_embeddings(id), axis=0)[None, :]], dim=0)

            else:
                input_ids = torch.stack(input_ids)
                dec_emb_in = self.encoder.embeddings.word_embeddings(input_ids)  # [batch, emb_dim]
            # add positional embeddings
            position_embeddings = self.position_embeddings(output[-1])
            dec_emb_in = dec_emb_in + position_embeddings
            dec_emb_in = self.LayerNorm(dec_emb_in)
            # concatenate two additional gaze features
            dec_in = torch.cat((dec_emb_in, torch.zeros(dec_emb_in.shape[0], 2).to(sn_emd.device)), dim=1)

        ####### duration prediction #######
        location_output = torch.stack(location_output, dim=0)  # [59, 256, 51]
        dec_emb_in_accumulated = torch.stack(dec_emb_in_accumulated, dim=0)  # [59, 256, 768]

        dur_emb_in = torch.cat([dec_emb_in_accumulated, location_output], dim=2)  # [59, 256, 819]
     #   print("dur_emb_in", dur_emb_in, dur_emb_in.shape)

        for j in range(dur_emb_in.shape[0]):
            hx_dur, cx_dur = self.decoder_duration_cell1(dur_emb_in[j], (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell2(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell3(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell4(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell5(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell6(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell7(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))
            hx_dur, cx_dur = self.decoder_duration_cell8(self.dropout_LSTM_duration(hx_dur), (hx_dur, cx_dur))

            # computes attention weights between the current hidden state of the fixation-sequence encoder (ht)
            # ad the output from the word-sequence encoder (hs).
            att_weight_dur = self.cross_attention(ht=hx_dur,  # current hidden state of the fixation-sequence encoder
                                                hs=enc_out,  # output from the word-sequence encoder
                                                sn_mask=sn_mask,
                                                cur_word_index=output[-1])

            context_dur = torch.matmul(att_weight_dur, enc_out)  # [256, 1, 129] [batch, 1, units]
            # Decoder
            hc_dur = torch.cat([context_dur.squeeze(1), hx_dur], dim=1)  # [256, 257] correct
            duration_result = self.decoder_duration(hc_dur)  # location is [256, 51]
            duration_output.append(duration_result)

        output = torch.stack(output, dim=0)  # [step, batch]  # torch.Size([60, 256])

        duration_output = torch.stack(duration_output, dim=0)  # torch.Size([256, 59, 1])
        duration_output = duration_output.squeeze(-1)
        print("duration_output", duration_output.permute(1, 0), duration_output.permute(1, 0).shape)

        # Concatenate the zero tensor to duration_output along dimension 0 (rows)
        #	zero_tensor = torch.zeros(1, 256).to(duration_output.device)
        #	duration_output = torch.cat([zero_tensor, duration_output], dim=0)

        #	density_prediction = torch.stack(density_prediction, dim=0)

        #	print("duration output", duration_output.permute(1,0), duration_output.permute(1,0).shape)  # [256, 60]
        print("location output", output.permute(1, 0), output.permute(1, 0).shape)  # [256, 60] , which is correct
        #	print("density_prediction", density_prediction, density_prediction.shape) # torch.Size([59, 256, 51])
        return output.permute(1, 0), density_prediction, duration_output


class Eyettention_readerID(nn.Module):
    def __init__(self, cf):
        super(Eyettention_readerID, self).__init__()
        self.cf = cf
        self.window_width = 1  # D
        self.atten_type = cf["atten_type"]
        self.hidden_size = 128
        self.sub_emb_size = cf["subid_emb_size"]

        # Word-Sequence Encoder
        encoder_config = BertConfig.from_pretrained(self.cf["model_pretrained"])
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print("keeping Bert with pre-trained weights")
        self.encoder = BertModel.from_pretrained(self.cf["model_pretrained"], config=encoder_config)
        self.encoder.eval()
        # freeze the parameters in Bert model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(0.4)
        self.encoder_lstm = nn.LSTM(input_size=768,  # BERT embedding size
									hidden_size=int(self.hidden_size / 2),
									num_layers=8,
									batch_first=True,
									bidirectional=True,
									dropout=0.2)

        # Fixation-Sequence Encoder
        self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
        # for reader-specific setting
        self.sub_embeddings = nn.Embedding(400, self.sub_emb_size)

        # The scanpath is generated in an autoregressive manner, the output of the previous timestep is fed to the input of the next time step.
        # So we use decoder cells and loop over all timesteps.
        # initialize eight decoder cells
        self.decoder_cell1 = nn.LSTMCell(768 + 2 + self.sub_emb_size,
                                         self.hidden_size)  # first layer input size = #BERT embedding size + two fixation attributes:landing position and fixiation duration
        self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.dropout_LSTM = nn.Dropout(0.2)

        # Cross-Attention
        self.attn = nn.Linear(self.hidden_size, self.hidden_size + 1)  # +1 acoount for the word length feature

        # Decoder
        # initialize five dense layers
        self.dropout_dense = nn.Dropout(0.2)
        self.decoder_dense = nn.Sequential(
            self.dropout_dense,
            nn.Linear(self.hidden_size * 2 + 1, 512),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            self.dropout_dense,
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cf["max_sn_len"] * 2 - 3),  # number of output classes
        )

        # for scanpath generation
        self.softmax = nn.Softmax(dim=1)

    def pool_subword_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
        # batching computing
        # Pool bert token (subword) to word level
        if target == 'sn':
            max_len = self.cf["max_sn_len"]  # CLS and SEP included
        elif target == 'sp':
            max_len = self.cf["max_sp_len"] - 1  # do not account the 'SEP' token

        merged_word_emb = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
        for word_idx in range(max_len):
            word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
            # pooling method -> sum
            if pool_method == 'sum':
                pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            elif pool_method == 'mean':
                pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1)  # [batch, 1, 768]
            merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

        mask_word = torch.sum(merged_word_emb, 2).bool()
        return merged_word_emb, mask_word

    def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
        # Word-Sequence Encoder
        outputs = self.encoder(input_ids=sn_emd, attention_mask=sn_mask)
        hidden_rep_orig, pooled_rep = outputs[0], outputs[1]
        if word_ids_sn != None:
            # Pool bert subword to word level for english corpus
            merged_word_emb, sn_mask_word = self.pool_subword_to_word(hidden_rep_orig,
                                                                      word_ids_sn,
                                                                      target='sn',
                                                                      pool_method='sum')
        else:  # no pooling for Chinese corpus
            merged_word_emb, sn_mask_word = hidden_rep_orig, None

        hidden_rep = self.embedding_dropout(merged_word_emb)
        x, (hn, hc) = self.encoder_lstm(hidden_rep, None)

        # concatenate with the word length feature
        x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
        return x, sn_mask_word

    def cross_attention(self, ht, hs, sn_mask, cur_word_index):
        # General Attention:
        # score(ht,hs) = (ht^T)(Wa)hs
        # hs is the output from word-Sequence Encoder
        # ht is the previous hidden state from Fixation-Sequence Encoder
        # self.attn(o): [batch, step, units]
        attn_prod = torch.matmul(self.attn(ht.unsqueeze(1)), hs.permute(0, 2, 1))  # [batch, 1, step]
        if self.atten_type == 'global':  # global attention
            attn_prod += (~sn_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

        else:  # local attention
            # current fixated word index
            aligned_position = cur_word_index
            # Get window borders
            left = torch.where(aligned_position - self.window_width >= 0, aligned_position - self.window_width, 0)
            right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"] - 1,
                                aligned_position + self.window_width, self.cf["max_sn_len"] - 1)

            # exclude padding tokens
            # only consider words in the window
            sen_seq = torch.arange(self.cf["max_sn_len"])[None, :].expand(sn_mask.shape[0], self.cf["max_sn_len"]).to(
                sn_mask.device)
            outside_win_mask = (sen_seq < left.unsqueeze(1)) + (sen_seq > right.unsqueeze(1))

            attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
            att_weight = softmax(attn_prod, dim=2)  # [batch, 1, step]

            if self.atten_type == 'local-g':  # local attention with Gaussian Kernel
                gauss = lambda s: torch.exp(-torch.square(s - aligned_position.unsqueeze(1)) / (
                            2 * torch.square(torch.tensor(self.window_width / 2))))
                gauss_factor = gauss(sen_seq)
                att_weight = att_weight * gauss_factor.unsqueeze(1)

        return att_weight

    def decode(self, sp_emd, sn_mask, sp_pos, enc_out, sp_fix_dur, sp_landing_pos, word_ids_sp, sub_id):
        # Fixation-Sequence Encoder + Decoder
        # Initialize hidden state and cell state with zeros
        hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
        hx, cx = hn[0, :, :], hc[0, :, :]
        hx2, cx2 = hn[1, :, :], hc[1, :, :]
        hx3, cx3 = hn[2, :, :], hc[2, :, :]
        hx4, cx4 = hn[3, :, :], hc[3, :, :]
        hx5, cx5 = hn[4, :, :], hc[4, :, :]
        hx6, cx6 = hn[5, :, :], hc[5, :, :]
        hx7, cx7 = hn[6, :, :], hc[6, :, :]
        hx8, cx8 = hn[7, :, :], hc[7, :, :]

        dec_emb_in = self.encoder.embeddings.word_embeddings(sp_emd[:, :-1])
        if word_ids_sp is not None:
            # Pool bert subword to word level for English corpus
            sp_merged_word_emd, sp_mask_word = self.pool_subword_to_word(dec_emb_in,
                                                                         word_ids_sp[:, :-1],
                                                                         target='sp',
                                                                         pool_method='sum')
        else:  # no pooling for Chinese
            sp_merged_word_emd, sp_mask_word = dec_emb_in, None

        # add positional embeddings and layer normalization
        position_embeddings = self.position_embeddings(sp_pos[:, :-1])
        dec_emb_in = sp_merged_word_emd + position_embeddings
        dec_emb_in = self.LayerNorm(dec_emb_in)
        dec_emb_in = dec_emb_in.permute(1, 0, 2)  # [step, n, emb_dim]
        dec_emb_in = self.embedding_dropout(dec_emb_in)

        # concatenate two additional gaze features
        if sp_landing_pos is not None:
            dec_emb_in = torch.cat((dec_emb_in, sp_landing_pos.permute(1, 0)[:-1, :, None]), dim=2)

        if sp_fix_dur is not None:
            dec_emb_in = torch.cat((dec_emb_in, sp_fix_dur.permute(1, 0)[:-1, :, None]), dim=2)

        # concatenate subject id for Eyettention_reader setting
        if sub_id is not None:
            dec_emb_in = torch.cat((dec_emb_in, self.sub_embeddings(sub_id).repeat(dec_emb_in.shape[0], 1, 1)), dim=2)

        # Predict output for each time step in turn
        output = []
        # save attention scores for visualization
        atten_weights_batch = torch.empty(sp_emd.shape[0], 0, self.cf["max_sn_len"]).to(sp_emd.device)
        for i in range(dec_emb_in.shape[0]):
            hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))  # [batch, units]
            hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
            hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
            hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(hx3), (hx4, cx4))
            hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(hx4), (hx5, cx5))
            hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(hx5), (hx6, cx6))
            hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(hx6), (hx7, cx7))
            hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(hx7), (hx8, cx8))

            att_weight = self.cross_attention(ht=hx8,
                                              hs=enc_out,
                                              sn_mask=sn_mask,
                                              cur_word_index=sp_pos[:, i])
            atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)

            context = torch.matmul(att_weight, enc_out)  # [batch, 1, units]

            # Decoder
            hc = torch.cat([context.squeeze(1), hx8], dim=1)  # [batch, units *2]
            result = self.decoder_dense(hc)  # [batch, dec_o_dim]
            output.append(result)

        output = torch.stack(output, dim=0)  # [step, batch, dec_o_dim]
        # output = F.softmax(output, dim=2) # cross entropy in pytorch includes softmax
        return output.permute(1, 0, 2), atten_weights_batch  # [batch, step, dec_o_dim]

    def forward(self, sn_emd, sn_mask, sp_emd, sp_pos, word_ids_sn, word_ids_sp, sp_fix_dur, sp_landing_pos,
                sn_word_len, sub_id):
        x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)  # [batch, step, units], [batch, units]

        if sn_mask_word is None:  # for Chinese dataset without token pooling
            sn_mask = torch.Tensor.bool(sn_mask)
            pred, atten_weights = self.decode(sp_emd,
                                              sn_mask,
                                              sp_pos,
                                              x,
                                              sp_fix_dur,
                                              sp_landing_pos,
                                              word_ids_sp,
                                              sub_id)  # [batch, step, dec_o_dim]

        else:  # for English dataset with token pooling
            pred, atten_weights = self.decode(sp_emd,
                                              sn_mask_word,
                                              sp_pos,
                                              x,
                                              sp_fix_dur,
                                              sp_landing_pos,
                                              word_ids_sp,
                                              sub_id)  # [batch, step, dec_o_dim]
        return pred, atten_weights

