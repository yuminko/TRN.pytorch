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
        self.loss_method = args.loss_method

        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        
        self.lstm_oad = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.lstm_delta = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.classifier_oad = nn.Linear(self.hidden_size, self.num_classes)
        self.classifier_delta = nn.Linear(self.hidden_size, self.num_classes)
        self.classifier_deltav = nn.Linear(self.hidden_size, self.num_classes)

        self.weight = nn.Parameter(torch.randn(self.num_classes, self.hidden_size))

        self.softplus = nn.Softplus()

    def encoder(self, camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, enc_score, delta, test=False):
        before_score = enc_score

        fusion_input = self.feature_extractor(camera_input, sensor_input)

        enc_hx, enc_cx = self.lstm_oad(fusion_input, (enc_hx, enc_cx))
        d_enc_hx, d_enc_cx = self.lstm_delta(fusion_input, (d_enc_hx, d_enc_cx))       # [32,4096] 

        ## weighted embedding
        # print(before_score.sum(1))
        before_score = before_score.unsqueeze(2)                #[32,22,1] 
        before_score = before_score * self.weight               #[32,22, 4096]
        before_score = torch.sum(before_score, dim=1)           #[32,4096]
        hx_enc = torch.add(d_enc_hx, before_score)              #[32,4096]

        # new_enc_hx = torch.add(enc_hx, before_score)
        # enc_score = self.classifier_oad(new_enc_hx)

        enc_score = self.classifier_oad(enc_hx)
        delta_score = self.classifier_delta(hx_enc)
        delta_var = self.classifier_deltav(hx_enc)

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

        if self.loss_method == 'state_before':
            var = dist.variance    
            enc_var = [torch.diag(var_i) for var_i in var]      #(32,22,22)
            enc_var = torch.stack(enc_var, dim=0)

            delta_var = norm_dist.covariance_matrix             #(32,22,22)
            
    
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
                    var = dist.variance
                    # print('OAD variance')
                    # print(var.cpu().numpy()[0])
                    enc_diagonal = np.diag(var.cpu().numpy()[0])
                    
                    return enc_hx, enc_cx, enc_score, enc_diagonal
                    
                elif delta == True:
                    # print('DELTA')
                    delta_socre = norm_dist.mean
                    delta_vari = norm_dist.variance
                    delta_cov = np.diag(delta_vari.cpu().numpy()[0])
                    # print('DELTA variance')
                    # print(delta_vari.cpu().numpy()[0] )
                    # delta_cov = norm_dist.covariance_matrix
                    # delta_cov = delta_cov.reshape((22,22))
                    return d_enc_hx, d_enc_cx, delta_score, delta_cov
        
        if self.loss_method == 'oad_before':
            return enc_hx, enc_cx, enc_score, d_enc_hx, d_enc_cx, delta_score
        
        elif self.loss_method == 'state_before' :
            return enc_hx, enc_cx, enc_score, enc_var, d_enc_hx, d_enc_cx, delta_score, delta_var





    def step(self, camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, enc_score, delta ):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score, enc_var = \
                self.encoder(camera_input, sensor_input, enc_hx, enc_cx, d_enc_hx, d_enc_cx, enc_score, delta=delta, test=True)

        return enc_hx, enc_cx, enc_score, enc_var



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        d_enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        d_enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))

        enc_score = camera_inputs.new_zeros((batch_size,self.num_classes))

        oad_score_stack = []
        delta_score_stack = []

        oad_var_stack = []
        delta_var_stack = []

        # for _ in range(2):
        delta_score_stack.append(enc_score) ### for delta path = t-1

        if self.loss_method == 'oad_before':

            for enc_step in range(self.enc_steps):
                enc_hx, enc_cx, enc_score, d_enc_hx, d_enc_cx, delta_score = self.encoder(
                    camera_inputs[:, enc_step],
                    sensor_inputs[:, enc_step], enc_hx, enc_cx, d_enc_hx, d_enc_cx, enc_score, delta=False
                )

                oad_score_stack.append(enc_score)
                delta_score_stack.append(delta_score)
            
            delta_score_stack = delta_score_stack[0:-1]

            oad_score = torch.stack(oad_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)
            delta_score = torch.stack(delta_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)

            return oad_score, delta_score

        elif self.loss_method == 'state_before':

            for enc_step in range(self.enc_steps):
                enc_hx, enc_cx, enc_score, enc_var, d_enc_hx, d_enc_cx, delta_score, delta_var = self.encoder(
                    camera_inputs[:, enc_step],
                    sensor_inputs[:, enc_step], enc_hx, enc_cx, d_enc_hx, d_enc_cx, enc_score, delta=False
                )

                oad_score_stack.append(enc_score)
                delta_score_stack.append(delta_score)
                oad_var_stack.append(enc_var)
                delta_var_stack.append(delta_var)
            
            delta_score_stack = delta_score_stack[0:-4]
            delta_var_stack = delta_var_stack[0:-4]

            oad_score = torch.stack(oad_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)
            delta_score = torch.stack(delta_score_stack, dim=1)  # (B, T, C) = (32, 64, 22)

            oad_var = torch.stack(oad_var_stack, dim=1)  # (B, T, C) = (32, 64, 22, 22)
            delta_var = torch.stack(delta_var_stack, dim=1)  # (B, T, C) = (32, 63, 22, 22) t-1

            return oad_score, delta_score, oad_var, delta_var


        