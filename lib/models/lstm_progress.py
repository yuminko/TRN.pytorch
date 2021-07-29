import torch
import torch.nn as nn
import torch.distributions as distributions

from .feature_extractor import build_feature_extractor



class progressLSTM(nn.Module):
    def __init__(self, args):
        super(progressLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.num_classes = args.num_classes
        
        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.enc_drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size + 1, self.num_classes)

        self.process_lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.progress_drop = nn.Dropout(0.1)
        self.process_classifier = nn.Linear(self.hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def encoder(self, camera_input, sensor_input, enc_hx, enc_cx, p_enc_hx, p_enc_cx ):

        fusion_input = self.feature_extractor(camera_input, sensor_input)
        p_enc_hx, p_enc_cx = self.process_lstm(fusion_input, (p_enc_hx, p_enc_cx))
        enc_hx, enc_cx = self.lstm(fusion_input, (enc_hx, enc_cx))

        p_enc_hx = self.progress_drop(p_enc_hx)
        enc_hx = self.enc_drop(enc_hx)

        process_score = self.process_classifier(p_enc_hx)
        # process_score = self.sigmoid(process_score)
        process_score = self.tanh(process_score)

        hx_enc = torch.cat([enc_hx, process_score],1) 
        enc_score = self.classifier(hx_enc)

        return enc_hx, enc_cx, enc_score, p_enc_hx, p_enc_cx, process_score,


    def step(self, camera_input, sensor_input, enc_hx, enc_cx, p_enc_hx, p_enc_cx):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score, p_enc_hx, p_enc_cx, process_score = \
                self.encoder(camera_input, sensor_input, enc_hx, enc_cx, p_enc_hx, p_enc_cx)

        return enc_hx, enc_cx, enc_score, p_enc_hx, p_enc_cx, process_score



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        p_enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        p_enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        score_stack = []
        process_stack = []

        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, enc_score, p_enc_hx, p_enc_cx, process_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step], enc_hx, enc_cx, p_enc_hx, p_enc_cx
            )
            score_stack.append(enc_score)
            process_stack.append(process_score)

        scores = torch.stack(score_stack, dim=1).view(-1, self.num_classes)
        process_scores = torch.stack(process_stack, dim=1).view(-1,1)
        # scores = torch.stack(score_stack, dim=1)    #(32,64,22)

        return scores, process_scores

        