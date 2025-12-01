import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        # policy head
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        # value head
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 1)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, new_action = None):
        # x = x.float() / 255.
        # x = self.cnn(x)
        # x = torch.flatten(x, start_dim=1)
        hidden = self.cnn(x.float() / 255.)
        hidden = torch.flatten(hidden, start_dim=1)
        
        value = self.value(hidden)
        value = torch.squeeze(value)

        logits = self.action_logits(hidden)
        
        dist = Categorical(logits=logits)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy
        
        if eval:
            action = torch.argmax(logits, dim=1)
        else:
            if new_action is None:
                action = dist.sample()
            else:
                action = new_action

        # if new_action is None:
        #     action = dist.sample()
        
        
        # if eval is False:
        #     action = dist.sample()
        # else:
        #     if new_action is None:
        #         action = dist.sample()
        #     else:
        #         action = new_action
        
        return action, value, dist.log_prob(action), dist.entropy()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                