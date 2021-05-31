import torch
import torch.nn as nn
import torch.distributions as distributions

from .feature_extractor import build_feature_extractor



class SecondLSTM(nn.Module):
    def __init__(self, args):
        super(SecondLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.step_size = args.step_size
        self.num_classes = args.num_classes
        
        self.dirichlet = args.dirichlet
        self.method = args.method

        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        
        self.lstms = nn.ModuleList( [nn.LSTMCell(self.fusion_size, self.hidden_size) for i in range(len(self.step_size))])
        self.classifiers = nn.ModuleList([nn.Linear(self.hidden_size, self.num_classes) for i in range(len(self.step_size))])

        self.softplus = nn.Softplus()

    def encoder(self, fusion_input, enc_hx, enc_cx, lstm, classifier):
        # fusion_input = self.feature_extractor(camera_input, sensor_input)
        enc_hx, enc_cx = lstm(fusion_input, (enc_hx, enc_cx))
        enc_score = classifier(enc_hx)

        if self.dirichlet:
            if self.method == 'Mean':
                enc_score_soft = self.softplus(enc_score)
                dist = distributions.Dirichlet(enc_score_soft)
                enc_score = dist.mean

            elif self.method == 'Sample':
                enc_score_soft = self.softplus(enc_score)
                dist = distributions.Dirichlet(enc_score_soft)
                enc_score =dist.rsample()

        return enc_hx, enc_cx, enc_score


    def step(self, camera_input, sensor_input, enc_hx, enc_cx, step):
        # Encoder -> time t
        fusion_input = self.feature_extractor(camera_input, sensor_input)
        num = self.step_size.index(str(step))
        lstm = self.lstms[num]
        classifier = self.classifiers[num]
        enc_hx, enc_cx, enc_score = \
                self.encoder(fusion_input, enc_hx, enc_cx, lstm, classifier)

        return enc_hx, enc_cx, enc_score



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]

        dummy_score = camera_inputs.new_zeros((batch_size,self.num_classes))
        score_stacks = []

        for num, steps in enumerate(self.step_size):

            enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
            enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))

            each_step_score = []

            if steps !='0':
                for step in range(int(steps)):
                    each_step_score.append(dummy_score)   ### 처음에 dummy 추가

            for enc_step in range(self.enc_steps):
                fusion_input = self.feature_extractor(camera_inputs[:, enc_step], sensor_inputs[:,enc_step])
                lstm = self.lstms[num]
                classifier = self.classifiers[num]
                enc_hx, enc_cx, enc_score = self.encoder(
                    fusion_input, 
                    enc_hx, enc_cx, lstm, classifier)

                each_step_score.append(enc_score)


            if steps != '0':        
                each_step_score = each_step_score[0:-int(steps)]    ### delete 

            each_step_score = torch.stack(each_step_score, dim=1) # (B, T, C)
            score_stacks.append(each_step_score) # LEN_STEP * (B, T, C)
     

        # scores = torch.stack(score_stack, dim=1).view(-1, self.num_classes)
        # extend_scores = torch.stack(score_stack, dim=1).view(-1, self.enc_steps, self.num_classes)

        extend_scores = torch.stack(score_stacks, dim=0)  # (LEN_STEP, B, T, C)
        LEN_STEP, _, _, CHANNEL = extend_scores.size()
        scores = extend_scores.view(LEN_STEP, -1, CHANNEL) # (LEN_STEP, B*T, C)

        return scores, extend_scores

        