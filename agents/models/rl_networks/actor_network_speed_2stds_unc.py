import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init
from utilities.distributions import ContDist, SafeTruncatedNormal

class ActorNetwork(nn.Module):
    def __init__(self, state_size, fc_dims, lr, weight_decay, device, checkpoint_dir, log_sig_min=-20, log_sig_max=2, epsilon=1e-6):
        super(ActorNetwork, self).__init__()
        
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/weights/actor_network_speed.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/actor_network_speed.pt"
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.min_std = 0.1
        
        self.base = nn.Sequential(
            nn.Linear(state_size, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU())
        
        self.mean_linear = nn.Linear(fc_dims, 1)        # 뮤
        self.log_std_linear_rl = nn.Linear(fc_dims, 1)  # 시그마_RL
        self.std_linear_sl = nn.Linear(fc_dims, 1)      # 시그마_IL
        
        self.apply(weights_init)
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)

        
    def forward(self, state):
        x = self.base(state)
        # mean = torch.tanh(self.mean_linear(x))
        mean = self.mean_linear(x)
        log_std_rl = self.log_std_linear_rl(x)
        log_std_rl = torch.clamp(log_std_rl, min=self.log_sig_min, max=self.log_sig_max)
        std_sl = self.std_linear_sl(x)

        
        return mean, log_std_rl, std_sl
    
    def sample(self, state):
        mean, log_std, _ = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick (mean + std * N(0,1)).
        action = torch.tanh(x_t) # convert action into [-1,1].
        
        log_prob = normal.log_prob(x_t)
        
        # enforcing action bound. 
        log_prob -= torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
       
        
        return action, log_prob, mean, std

    def get_dist(self, state):
        # 💡 forward()에서 3개의 raw 값을 모두 받아옵니다.
        mean_raw, _, std_sl_raw = self.forward(state)
        
        # 💡 수정 1: std_sl_raw를 softplus를 통해 안정적으로 양수로 변환합니다.
        # 이렇게 하면 std가 음수가 되어 발생하는 런타임 에러를 원천적으로 방지할 수 있습니다.
        std = F.softplus(std_sl_raw) + self.min_std
        
        # 💡 수정 2: 분포를 생성할 때 mean_raw에 tanh를 적용합니다.
        # 이를 통해 분포의 중심점이 항상 [-1, 1] 범위 내에 있도록 보장하여,
        # SafeTruncatedNormal 분포의 수치적 안정성을 극대화합니다.
        dist = SafeTruncatedNormal(torch.tanh(mean_raw), std, -1, 1)
        dist = ContDist(torch.distributions.independent.Independent(dist, 1))
        
        return dist
        
    # def save_checkpoint(self):
    #     # === 안전망: 저장 전에 폴더 보장 ===
    #     os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
    #     os.makedirs(os.path.dirname(self.checkpoint_optimizer), exist_ok=True)
    #     torch.save(self.state_dict(), self.checkpoint_file)
    #     torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
        
    def save_checkpoint(self, directory=None, suffix=''):
        if directory is None:
            # 기본 저장 경로 (Best model)
            weights_dir = os.path.join(self.checkpoint_dir, 'weights')
            optimizer_dir = os.path.join(self.checkpoint_dir, 'weights', 'optimizers')
        else:
            # 주기적 저장 경로
            weights_dir = directory
            optimizer_dir = os.path.join(directory, 'optimizers')

        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(optimizer_dir, exist_ok=True)

        base_model_filename = "actor_network_speed"
        base_optimizer_filename = "actor_network_speed"

        model_filepath = os.path.join(weights_dir, f"{base_model_filename}{suffix}.pt")
        optimizer_filepath = os.path.join(optimizer_dir, f"{base_optimizer_filename}{suffix}.pt")

        torch.save(self.state_dict(), model_filepath)
        torch.save(self.optimizer.state_dict(), optimizer_filepath)
    
    
    def load_checkpoint(self):
        # === 안전망: 로드 전에 폴더 보장 ===
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_optimizer), exist_ok=True)
        # 파일이 없으면 조용히 건너뜀(평가만 먼저 돌릴 때 크래시 방지)
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        if os.path.exists(self.checkpoint_optimizer):
            # optimizer state를 cpu로 먼저 load
            optimizer_state = torch.load(self.checkpoint_optimizer, map_location='cpu')
            # 그 다음, optimizer에 state를 load
            self.optimizer.load_state_dict(optimizer_state)
        
        
    # def load_checkpoint(self, directory=None, suffix=''):
    #     if directory is None:
    #         weights_dir = os.path.join(self.checkpoint_dir, 'weights')
    #         optimizer_dir = os.path.join(self.checkpoint_dir, 'weights', 'optimizers')
    #     else:
    #         weights_dir = directory
    #         optimizer_dir = os.path.join(directory, 'optimizers')
            
    #     base_model_filename = "actor_network_speed"
    #     base_optimizer_filename = "actor_network_speed"

    #     model_filepath = os.path.join(weights_dir, f"{base_model_filename}{suffix}.pt")
    #     optimizer_filepath = os.path.join(optimizer_dir, f"{base_optimizer_filename}{suffix}.pt")
        
    #     if os.path.exists(model_filepath):
    #         self.load_state_dict(torch.load(model_filepath, map_location=self.device))
    #     if os.path.exists(optimizer_filepath):
    #         optimizer_state = torch.load(optimizer_filepath, map_location='cpu')
    #         self.optimizer.load_state_dict(optimizer_state)
