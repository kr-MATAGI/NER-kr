import torch
import torch.nn as nn
import torch.nn.functional as F

#====================================================
class CharCNN(nn.Module):
#====================================================
    def __init__(self,
                 vocab_size,
                 seq_len
                 ):
        super(CharCNN, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = seq_len
        self.drop_prob = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.vocab_size, out_channels=256,
                kernel_size=(3,), stride=(1, ), padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)#, dilation=1, ceil_mode=False)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=256,
                kernel_size=(4,), stride=(1,)
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)#, dilation=1, ceil_mode=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=256,
                kernel_size=(5,), stride=(1,)
            ),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1280, 768),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
            x : [batch_size, seq_len * 3, vocab_size]
        '''

        # Conv Net
        x = self.conv1(x) # [batch_size, output_ch, vocab_size]
        # print("\nAAAAAA: ", x.shape, "\n")
        x = self.conv2(x) # [batch_size, output_ch, vocab_size]
        # print("\nBBBBBB: ", x.shape, "\n")
        x = self.conv3(x)
        # print("\nCCCCCC: ", x.shape, "\n")

        # Fully-Connected
        x = self.fc1(x.view(x.size(0), -1))
        x = self.log_softmax(x)

        return x