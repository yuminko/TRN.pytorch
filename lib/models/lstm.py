import torch
import torch.nn as nn
import torch.distributions as distributions

from .feature_extractor import build_feature_extractor



class baselineLSTM(nn.Module):
    def __init__(self, args):
        super(baselineLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.num_classes = args.num_classes
        
        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.weight = nn.Parameter(torch.randn(self.num_classes, self.hidden_size))
        self.softplus = nn.Softplus()

    def encoder(self, camera_input, sensor_input, enc_hx, enc_cx, enc_score):
        before_score = enc_score
        fusion_input = self.feature_extractor(camera_input, sensor_input)
        enc_hx, enc_cx = self.lstm(fusion_input, (enc_hx, enc_cx))

        before_score = before_score.unsqueeze(2)                #[32,22,1] 
        before_score = before_score * self.weight               #[32,22, 4096]
        before_score = torch.sum(before_score, dim=1)           #[32,4096]
        hx_enc = torch.add(enc_hx, before_score)

        enc_score = self.classifier(hx_enc)

        ## add dirichlet process
        enc_score_soft = self.softplus(enc_score)
        dist = distributions.Dirichlet(enc_score_soft)
        enc_score = dist.mean

        return enc_hx, enc_cx, enc_score


    def step(self, camera_input, sensor_input, enc_hx, enc_cx, enc_score):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score = \
                self.encoder(camera_input, sensor_input, enc_hx, enc_cx, enc_score)

        return enc_hx, enc_cx, enc_score



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        score_stack = []

        enc_score = camera_inputs.new_zeros((batch_size,self.num_classes))

        # print(enc_hx.shape)

        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, enc_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step], enc_hx, enc_cx, enc_score
            )
            score_stack.append(enc_score)
            # print(enc_score.shape)

        # print(len(score_stack))

        scores = torch.stack(score_stack, dim=1).view(-1, self.num_classes)
        # scores = torch.stack(score_stack, dim=1)    #(32,64,22)

        return scores

        