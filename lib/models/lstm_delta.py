import torch
import torch.nn as nn
import torch.distributions as distributions

import numpy as np

from .feature_extractor import build_feature_extractor



class DeltaLSTM(nn.Module):
    def __init__(self, args):
        super(DeltaLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.num_classes = args.num_classes
        
        self.dirichlet = args.dirichlet
        self.method = args.method
        self.var_method = args.var_method

        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        
        self.lstm_oad = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.lstm_delta = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.classifier_oad = nn.Linear(self.hidden_size, self.num_classes)
        self.classifier_delta = nn.Linear(self.hidden_size, self.num_classes)
        self.classifier_deltav = nn.Linear(self.hidden_size, self.num_classes)

        self.softplus = nn.Softplus()

    def encoder(self, camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, delta, test=False):
        fusion_input = self.feature_extractor(camera_input, sensor_input)

        if test:
            if self.var_method == 'covariance':
                con = dist.concentration
                con0 = con.sum(-1,True)
                d = (con0.pow(2) * (con0+1))
                con = con.cpu().numpy()[0]
                con_s = np.reshape(con, (22,1))
                con_t = np.reshape(con, (1,22))
                diagonal = -con_s * con_t 
                diagonal /= d.cpu().numpy()[0]
                var = dist.variance.cpu().numpy()[0]
                np.fill_diagonal(diagonal , var)


                return enc_hx, enc_cx, enc_score, diagonal
            
            elif self.var_method == 'diagonal':
                if delta == False:
                    enc_hx, enc_cx = self.lstm_oad(fusion_input, (enc_hx, enc_cx))
                    enc_score = self.classifier_oad(enc_hx)
                    enc_score_soft = self.softplus(enc_score)
                    dist = distributions.Dirichlet(enc_score_soft)
                    enc_score = dist.mean
                    var = dist.variance
                    enc_diagonal = np.diag(var.cpu().numpy()[0])
                    return enc_hx, enc_cx, enc_score, enc_diagonal
                elif delta == True:
                    d_enc_hx, d_enc_cx = self.lstm_delta(fusion_input, (d_enc_hx, d_enc_cx))
                    delta_score = self.classifier_delta(d_enc_hx)
                    delta_var = self.classifier_deltav(d_enc_hx)
                    delta_var_soft = self.softplus(delta_var)
                    diagonal = torch.diag(delta_var_soft.mean(dim=0))
                    diagonal = torch.tensor(diagonal)
                    norm_dist = distributions.MultivariateNormal(delta_score.mean(dim=0),diagonal)
                    delta_socre = norm_dist.mean
                    delta_vari = norm_dist.variance
                    delta_diagonal = np.diag(delta_vari.cpu().numpy())
                    return d_enc_hx, d_enc_cx, delta_score, delta_diagonal

        enc_hx, enc_cx = self.lstm_oad(fusion_input, (enc_hx, enc_cx))
        d_enc_hx, d_enc_cx = self.lstm_delta(fusion_input, (d_enc_hx, d_enc_cx))

        enc_score = self.classifier_oad(enc_hx)
        delta_score = self.classifier_delta(d_enc_hx)
        delta_var = self.classifier_deltav(d_enc_hx)

        if self.dirichlet:
            if self.method == 'Mean':
                enc_score_soft = self.softplus(enc_score)
                dist = distributions.Dirichlet(enc_score_soft)
                enc_score = dist.mean

            elif self.method == 'Sample':
                enc_score_soft = self.softplus(enc_score)
                dist = distributions.Dirichlet(enc_score_soft)
                enc_score =dist.rsample()

        ### diagonal 
        delta_var_soft = self.softplus(delta_var)
        diagonal = []
        for i in range(len(delta_var_soft)):
            diag = torch.diag(delta_var_soft[i])    
            diagonal.append(diag)
        diagonal = torch.stack(diagonal)
        norm_dist = distributions.MultivariateNormal(delta_score,diagonal)
        delta_score = norm_dist.rsample()

        return enc_hx, enc_cx, enc_score, d_enc_hx, d_enc_cx, delta_score


    def step(self, camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, delta ):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score, enc_var = \
                self.encoder(camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, delta=delta, test=True)

        return enc_hx, enc_cx, enc_score, enc_var



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        d_enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        d_enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))

        dummy_score = camera_inputs.new_zeros((batch_size,self.num_classes))

        oad_score_stack = []
        delta_score_stack = []

        delta_score_stack.append(dummy_score) ### for delta path = t-1

        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, enc_score, d_enc_hx, d_enc_cx, delta_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step], enc_hx, enc_cx, d_enc_hx, d_enc_cx, delta=False
            )

            oad_score_stack.append(enc_score)
            delta_score_stack.append(delta_score)
        
        delta_score_stack = delta_score_stack[0:-1]

        oad_score = torch.stack(oad_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)
        delta_score = torch.stack(delta_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)

        return oad_score, delta_score

        