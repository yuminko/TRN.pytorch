import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor



class SecondLSTM(nn.Module):
    def __init__(self, args):
        super(SecondLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.step_size = args.step_size
        self.num_classes = args.num_classes
        
        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size
        # print(self.fusion_size)
        
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def encoder(self, camera_input, sensor_input, enc_hx, enc_cx):
        fusion_input = self.feature_extractor(camera_input, sensor_input)
        # dummy_input = fusion_input.new_zeros((fusion_input.shape))
        enc_hx, enc_cx = self.lstm(fusion_input, (enc_hx, enc_cx))
        enc_score = self.classifier(enc_hx)
        return enc_hx, enc_cx, enc_score


    def step(self, camera_input, sensor_input, enc_hx, enc_cx):
        # Encoder -> time t
        enc_hx, enc_cx, enc_score = \
                self.encoder(camera_input, sensor_input, enc_hx, enc_cx)

        return enc_hx, enc_cx, enc_score



    def forward(self, camera_inputs, sensor_inputs):
        batch_size = camera_inputs.shape[0]
        enc_hx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cx = camera_inputs.new_zeros((batch_size, self.hidden_size))
        score_stack = []

        dummy_score = camera_inputs.new_zeros((batch_size,self.num_classes))

        for step in range(self.step_size):
            score_stack.append(dummy_score)   ### 처음에 dummy 추가

        for enc_step in range(self.enc_steps):
            enc_hx, enc_cx, enc_score = self.encoder(
                camera_inputs[:, enc_step],
                sensor_inputs[:, enc_step], enc_hx, enc_cx,
            )

            score_stack.append(enc_score)
        
        for step in range(self.step_size):
            del score_stack[-1]

        scores = torch.stack(score_stack, dim=1).view(-1, self.num_classes)

        return scores

        