import torch
import torch.nn as nn   ####### 변경점
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import os

from colorama import Fore
from importlib import import_module
from collections import deque

from agents.models.memory.memory import ReplayBufferStorage, make_replay_loader
from utilities.controls import carla_control, PID
from utilities.conversions import convert_11
from utilities.networks import update_target_network, RandomShiftsAug


class Agent():
    def __init__(self, training_config, augmentation_config, vehicle_measurements_config, waypoints_config, image_config, critic_config, actor_config, memory_config, control_config, maximum_speed, experiment_path, init_memory):

        self.maximum_speed = maximum_speed
        self.experiment_path = experiment_path
        self.alpha = training_config['alpha']
        self.automatic_entropy_tuning = training_config['automatic_alpha']
        self.device = torch.device(training_config['device'])
        self.batch_size = training_config['batch_size']
        self.discount_factor = training_config['discount_factor']
        self.state_size = image_config['out_dims'] + waypoints_config['out_dims'] + vehicle_measurements_config['out_dims']
        self.target_update_interval = training_config['target_update_interval']
        self.obs_info = self.parse_obs_info(
            memory_config['obs_info'])
        self.repeat_action = training_config['repeat_action']
        self.n_step = training_config['n_step']
        self.use_aug = augmentation_config['use_aug']
        self.critic_tau = critic_config['tau']
        self.num_waypoints = waypoints_config['num_waypoints']
        self.deque_size = training_config['deque_size']
        self.grad_clip = training_config['grad_clip']
        self.unc_threshold = training_config['unc_threshold']
        self.dem_used = 0.0
        self.image_size = image_config['image_size']

        os.makedirs(self.experiment_path, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_path, "weights"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_path, "weights", "optimizers"), exist_ok=True)
        
        ##### 변경점
        self.n_speed_heads = actor_config.get('n_speed_heads', 1) # 설정 파일에 없으면 기본값 1(기존 방식)
        self.bootstrap_p = actor_config.get('bootstrap_p', 1.0) # 설정 파일에 없으면 기본값 1.0(모든 데이터 보기)
        self.unc_threshold_epi = training_config.get('unc_threshold_epi', float('inf')) # 설정 파일에 없으면 전문가 개입 안함

        if self.use_aug:
            self.aug = RandomShiftsAug(pad=augmentation_config['pad'])
            self.aug2 = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.05), ratio=(0.1, 2.1))
                ])

        if init_memory:
            experiment_name = experiment_path.split('/')[-1]
            replay_dir = f"{os.getenv('HOME')}/memory/{experiment_name}"
            self.replay_storage = ReplayBufferStorage(
                obs_info=self.obs_info, replay_dir=replay_dir, n_actions=2)
            
            self.replay_loader = make_replay_loader(replay_dir=replay_dir, obs_info=self.obs_info, max_size=memory_config['capacity'], batch_size=self.batch_size, num_workers=memory_config['num_workers'], nstep=self.n_step, discount=self.discount_factor, deque_size=self.deque_size)
        
        # image encoder
        module_str, class_str = image_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.image_encoder = _Class(lr=training_config['lr'], weight_decay=training_config['weight_decay'], out_dims=image_config['out_dims'], checkpoint_dir=self.experiment_path, device=self.device, input_channels=self.deque_size*3)

        # waypoint encoder
        module_str, class_str = waypoints_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.waypoints_encoder = _Class(lr=training_config['lr'], num_waypoints=waypoints_config['num_waypoints'], fc_dims=waypoints_config['fc_dims'],
                                       out_dims=waypoints_config['out_dims'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path)

        # vehicle measurement encoder
        module_str, class_str = vehicle_measurements_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.vm_encoder = _Class(lr=training_config['lr'], num_inputs=vehicle_measurements_config['num_inputs'], fc_dims=vehicle_measurements_config['fc_dims'],
                                       out_dims=vehicle_measurements_config['out_dims'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path)

        # critic
        module_str, class_str = critic_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.critic = _Class(state_size=self.state_size, fc_dims=critic_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, target=False)

        self.critic_target = _Class(state_size=self.state_size, fc_dims=critic_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, target=True)

        # hard update using tau=1.
        update_target_network(self.critic_target, self.critic, tau=1)

        self.pid = PID(kp=control_config['pid']['kp'], ki=control_config['pid']['ki'],
                       kd=control_config['pid']['kd'], dt=control_config['pid']['dt'], maximum_speed=maximum_speed)

        # actor
        module_str, class_str = actor_config['entry_point_steer'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.policy_steer = _Class(state_size=self.state_size, fc_dims=actor_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, log_sig_min=actor_config['log_sig_min'], log_sig_max=actor_config['log_sig_max'], epsilon=actor_config['epsilon'])

        # actor
        module_str, class_str = actor_config['entry_point_speed'].split(':')
        _Class = getattr(import_module(module_str), class_str)

        #변경점
        speed_heads = []
        for i in range(self.n_speed_heads):
            # 💡 --- 오류 수정 ---
            # 각 Head가 사용할 폴더 경로를 정의하고, 폴더가 없으면 생성해줍니다.
            head_checkpoint_dir = os.path.join(self.experiment_path, f"head_{i}")
            os.makedirs(head_checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(head_checkpoint_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(head_checkpoint_dir, "weights", "optimizers"), exist_ok=True)
            
            head = _Class(
                state_size=self.state_size, 
                fc_dims=actor_config['fc_dims'], 
                lr=training_config['lr'], 
                weight_decay=training_config['weight_decay'], 
                device=self.device, 
                checkpoint_dir=head_checkpoint_dir, # 생성된 폴더 경로를 전달
                log_sig_min=actor_config['log_sig_min'], 
                log_sig_max=actor_config['log_sig_max'], 
                epsilon=actor_config['epsilon']
            )
            speed_heads.append(head)
        self.policy_speed = nn.ModuleList(speed_heads)
        
        # 아래 기존 코드
        # self.policy_speed = _Class(state_size=self.state_size, fc_dims=actor_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, log_sig_min=actor_config['log_sig_min'], log_sig_max=actor_config['log_sig_max'], epsilon=actor_config['epsilon'])

        if self.automatic_entropy_tuning:
            self.target_entropy = - \
                torch.prod(torch.Tensor([2]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(training_config['alpha']), requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=training_config['lr_alpha'])
 
 
        # init vars.
        self.action_ctn = 0
        self.prev_action = None
        self.train_ctn = 0
        self._replay_iter = None

    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @staticmethod
    def parse_obs_info(obs_info):
        for state_key, state_value in obs_info.items():
            for key, value in state_value.items():
                obs_info[state_key][key] = eval(value)
        return obs_info

    def encode(self, obs, detach=False):
        image = obs['image'] 
        waypoints = obs['waypoints']  
        vm = obs['vehicle_measurements']

        state_image = self.image_encoder(image)
        state_waypoints = self.waypoints_encoder(waypoints)
        state_vm = self.vm_encoder(vm)
            
        if detach:
            state_image = state_image.detach()
            state_waypoints = state_waypoints.detach()
            state_vm = state_vm.detach()

        state = torch.cat([state_image, state_waypoints, state_vm], dim=1) 

        return state

    @torch.no_grad()
    ### 변경점 : 아래 choose_action 함수 (obs, obs_torch 저렇게 사용하는 거 맞는지 의문)
    def choose_action(self, obs, step, training=True):
        """
        [수정됨] Post-tanh Averaging & Uncertainty Calculation 적용
        - 관측을 토치 텐서로 만들 때 항상 배치차원(B)을 붙여 [B, ...] 유지
        - speed/steer는 반드시 [B,1] 모양으로 강제
        - Uncertainty는 Tanh를 거친 Action Space에서 계산하여 Scale 문제 해결
        """
        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:
            # 1) 관측 전처리 + deque 업데이트
            obs_filtered = self.filter_obs(obs=obs)
            self.update_deque(obs=obs_filtered)

            # 2) 토치 텐서로 변환 (B차원 포함)
            obs_torch = self.convert_obs_into_torch(obs=obs_filtered, unsqueeze=True)

            # 3) 인코딩 (액션 선택 경로는 detach=True)
            state = self.encode(obs=obs_torch, detach=True)             # [B, state_size]

            # 4) 조향(steer): 기존 로직 유지하되 모양을 [B,1]로 강제
            if training:
                steer, _, _ = self.policy_steer.sample(state)           # [B,1]
            else:
                _, _, steer = self.policy_steer.sample(state)           # [B,1]
            steer = steer.view(-1, 1)                                   # 보수적으로 [B,1] 고정

            # 5) 속도(speed) 앙상블: 각 head의 pre-tanh μ와 RL std 수집
            mus_logit_list, sig2s = [], []
            for i in range(self.n_speed_heads):
                # sample() = (action[tanh], logp, mean[pre-tanh], std_rl)
                _, _, mu_logit_i, std_rl_i = self.policy_speed[i].sample(state)  # [B,1] 각각
                mus_logit_list.append(mu_logit_i)                        # [B,1] (Logit)
                sig2s.append(std_rl_i ** 2)                              # 분산 σ_RL^2, [B,1]

            # [H,B,1] -> squeeze(-1)로 [H,B]
            mus_logit = torch.stack(mus_logit_list, dim=0).squeeze(-1)   # [H,B] (Logit Space)
            sig2 = torch.stack(sig2s, dim=0).squeeze(-1)                 # [H,B]

            # 💡 [핵심 변경점 1] 먼저 Tanh를 적용하여 Action Space로 변환
            # 이유: Logit Space의 분산은 고속 주행 시(Saturation) 과도하게 커지는 문제가 있음.
            #       실제 물리적 행동의 차이를 불확실성으로 정의하기 위함.
            mus_action = torch.tanh(mus_logit)                           # [H,B] (Action Space: -1~1)

            # 💡 [핵심 변경점 2] Epistemic Uncertainty를 Action Space 분산으로 계산
            sigma_epi = mus_action.var(dim=0, unbiased=False).sqrt()     # [B]
            
            # Aleatoric Uncertainty (기존 유지 - Variance들의 평균)
            sigma_ale = (sig2.mean(dim=0)).sqrt()                        # [B]

            if step % 1000 == 0:
                print(f"[Step {step:07d}] Ale: {sigma_ale.mean().item():.4f} | Epi: {sigma_epi.mean().item():.4f}")
            
            # 배치=1 가정하에 스칼라 기준으로 게이트 판단
            gate = (sigma_ale.mean().item() > self.unc_threshold) or \
                   (sigma_epi.mean().item() > self.unc_threshold_epi)

            # 6) 속도 선택
            if training:
                # --- 훈련 시 최종 속도 ---
                if gate:
                    # --- 전문가 속도 사용(IL): 항상 [B,1]로 강제 ---
                    sp = convert_11(obs_filtered['desired_speed'])  # [-1,1] 범위
                    if isinstance(sp, torch.Tensor):
                        speed = sp.to(self.device).view(1, 1)               # [1,1]
                    elif isinstance(sp, np.ndarray):
                        speed = torch.from_numpy(sp).float().to(self.device).view(1, 1)
                    else:
                        speed = torch.tensor(sp, dtype=torch.float32, device=self.device).view(1, 1)
                    self.dem_used = 1.0
                else:
                    # --- RL: 랜덤 head 하나에서 샘플 (탐색) ---
                    idx = random.randint(0, self.n_speed_heads - 1)
                    # 여기서도 이미 내부적으로 tanh 처리된 action이 나옴
                    speed, _, _, _ = self.policy_speed[idx].sample(state)   # [B,1]
                    speed = speed.view(-1, 1)                                # [B,1] 보수적 고정
                    self.dem_used = 0.0
            else:
                # 💡 [핵심 변경점 3] 평가(Evaluation) 시: Action들의 평균 사용 (Post-tanh Mean)
                # 기존: speed = torch.tanh(mus_logit.mean(dim=0)) -> Outlier에 취약함
                # 변경: 실제 제안된 행동들의 평균을 사용하여 안전성 확보
                speed = mus_action.mean(dim=0).view(-1, 1)               # [B,1]

            # 7) 최종 액션 결합: [B,2] → numpy (B=1 가정)
            speed = speed.to(self.device).view(-1, 1)                        # 안전고정
            steer = steer.to(self.device).view(-1, 1)                        # 안전고정
            action = torch.cat([speed, steer], dim=1).detach().cpu().numpy()[0]  # (2,)
        else:
            action = self.prev_action  # 이전 액션 재사용

        # action 값에 NaN이나 inf가 포함되어 있는지 확인하고, 있다면 안전한 값(0.0)으로 대체
        if np.isnan(action).any() or np.isinf(action).any():
            print(f"{Fore.RED}!! Invalid action detected: {action}. Overwriting with zeros. !!{Fore.RESET}")
            action = np.zeros_like(action)

        # 8) CARLA control 변환 및 상태 업데이트
        controls = carla_control(self.pid.get(action=action, velocity=current_velocity))
        self.action_ctn += 1
        self.prev_action = action
        return action, controls

    def random_action(self, obs):

        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:

            action = np.asarray([random.uniform(-1, 1), random.uniform(-1, 1)])
        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def filter_obs(self, obs):
        obs_ = {}

        resized_image = cv2.resize(
            obs['image']['data'], self.obs_info['image']['shape'][1:3], interpolation=cv2.INTER_AREA)

        obs_['image'] = np.einsum(
            'kij->jki', resized_image)

        obs_['waypoints'] = np.array(
            obs['waypoints']['location'])[0:self.num_waypoints, 0:2].reshape(self.num_waypoints, 2)

        obs_speed = np.array(
            obs['speed']['speed'][0] / self.maximum_speed, dtype=np.float32).reshape(1)

        obs_steer = np.array(
            obs['control']['steer'][0]).reshape(1)

        obs_['vehicle_measurements'] = np.concatenate([obs_speed, obs_steer]).reshape(2)
        
        obs_['desired_speed'] = np.array(
            obs['desired_speed'] / self.maximum_speed, dtype=np.float32).reshape(1)


        return obs_

    def update_deque(self, obs):
        self.img_deque.append(obs['image'])
        obs['image'] = np.concatenate(list(self.img_deque), axis=0)

    def convert_obs_into_torch(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = torch.from_numpy(
                    value).to(self.device).unsqueeze(0)
            else:
                obs[key] = torch.from_numpy(value).to(self.device)
        return obs

    def convert_obs_into_device(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = value.to(self.device).unsqueeze(0)
            else:
                obs[key] = value.to(self.device)
        return obs

    def remember(self, obs, action, reward, next_obs, done):
        obs = None
        next_obs = self.filter_obs(next_obs)
        self.replay_storage.add(action=action, reward=reward, next_obs=next_obs, done=done)

    def augment_obs(self, obs):
        img = self.aug(obs['image'].float()) / 255.
        img = img.view(self.batch_size * self.deque_size, 3, self.image_size, self.image_size)
        img = self.aug2(img)
        img = img.view(self.batch_size, 3*self.deque_size, self.image_size, self.image_size) * 255.
        obs['image'] = img
        
        return obs

    def clone_obs(self, obs):
        obs_ = {}
        for key, value in obs.items():
            obs_[key] = value.clone()

        return obs_

    def train(self, step):
        self.train_ctn += 1

        metrics = dict()
        
        if self.train_ctn < 1024:
            return metrics

        # sample batch from memory.
        obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch = tuple(
            next(self.replay_iter))

        obs_batch = self.convert_obs_into_device(
            obs_batch)
        next_obs_batch = self.convert_obs_into_device(
            next_obs_batch)

        if self.use_aug:
            obs_batch = self.augment_obs(
                obs=obs_batch)
            next_obs_batch = self.augment_obs(
                obs=next_obs_batch)
        
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        discount_batch = discount_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # critic networks.
        metrics.update(self.update_critics(
                obs_batch=obs_batch, action_batch=action_batch, reward_batch=reward_batch, discount_batch=discount_batch,
                next_obs_batch=next_obs_batch, done_batch=done_batch))
        
        # actor networks.
        metrics.update(self.update_policy(obs_batch=obs_batch))
        metrics.update(self.update_policy_speed(obs_batch=obs_batch))
        
        metrics['dem_used'] = self.dem_used

        if self.train_ctn % self.target_update_interval == 0:
            update_target_network(target=self.critic_target,
                                  source=self.critic, tau=self.critic_tau)
            
        return metrics

    def update_critics(self, obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch):
        metrics = dict()
        
        with torch.no_grad():
            next_state_batch = self.encode(next_obs_batch)
            # [FIX] speed head 하나 선택
            head_idx = random.randint(0, self.n_speed_heads - 1)
            next_speed, next_logp_speed, _, _ = self.policy_speed[head_idx].sample(next_state_batch)
            next_steer, next_logp_steer, _ = self.policy_steer.sample(next_state_batch)

            next_action   = torch.cat([next_speed, next_steer], dim=1)
            next_log_prob = next_logp_speed + next_logp_steer

            q1_t, q2_t = self.critic_target(next_state_batch, next_action)
            min_q_next_target = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            q_value_target = reward_batch + (discount_batch * min_q_next_target)

        state_batch = self.encode(obs_batch)
        # two q-functions to mitigate positive bias in the policy update step.
        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_value_target)
        q2_loss = F.mse_loss(q2, q_value_target)
        q_loss = q1_loss + q2_loss

        self.critic.optimizer.zero_grad(set_to_none=True)
        self.image_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoints_encoder.optimizer.zero_grad(set_to_none=True)
        self.vm_encoder.optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.waypoints_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.vm_encoder.parameters(), self.grad_clip)
        

        self.critic.optimizer.step()
        self.image_encoder.optimizer.step()
        self.waypoints_encoder.optimizer.step()
        self.vm_encoder.optimizer.step()
        
        metrics['critic_loss'] = round(q_loss.item(), 4)


        return metrics

    ### 변경점 : 아래 update_policy 함수
    def update_policy(self, obs_batch):
        metrics = dict()
        
        state_batch = self.encode(obs_batch, detach=True)
        
        # 🆕 --- RL 손실: 무작위 Speed Actor 1개 선택 ---
        chosen_head_idx_rl = random.randint(0, self.n_speed_heads - 1)
        actions_speed, log_prob_speed, _, _ = self.policy_speed[chosen_head_idx_rl].sample(state_batch)
        
        actions_steer, log_prob_steer, _ = self.policy_steer.sample(state_batch)
        
        actions = torch.cat([actions_speed, actions_steer], dim=1)
        log_prob =  log_prob_speed + log_prob_steer
        
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        policy_loss = ((self.alpha * log_prob) - min_q).mean()

        # 🆕 --- 옵티마이저 업데이트 수정: RL 손실은 선택된 Head와 Steer만 업데이트 ---
        self.policy_speed[chosen_head_idx_rl].optimizer.zero_grad(set_to_none=True)
        self.policy_steer.optimizer.zero_grad(set_to_none=True)
        
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_speed[chosen_head_idx_rl].parameters(), self.grad_clip)
        self.policy_speed[chosen_head_idx_rl].optimizer.step()
        
        torch.nn.utils.clip_grad_norm_(self.policy_steer.parameters(), self.grad_clip)
        self.policy_steer.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob +
                           self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            alpha_logs = self.alpha.clone()
            

        metrics['policy_loss'] = round(policy_loss.item(), 4)
        metrics['alpha_loss'] = round(alpha_loss.item(), 4)
        metrics['alpha_logs'] = round(alpha_logs.item(), 4)
        

        return metrics 

    ### 변경점 : 아래 update_policy_speed 함수
    def update_policy_speed(self, obs_batch):
        """
        - IL(속도) 업데이트 경로
        - ground_truth(desired_speed)를 항상 torch.float32, [B,1], device= self.device 로 정규화
        - log_prob 출력(대개 [B])을 [B,1]로 맞춰 mask([B,1])과 동일 차원에서 연산
        """
        metrics = dict()

        # 1) 인코딩: IL에서는 인코더에도 그라디언트가 흐르도록 detach=False
        state_batch = self.encode(obs_batch)  # [B, state_size]

        # 2) GT 속도 준비: convert_11 -> torch.float32 on device -> [B,1]
        gt_tensor = obs_batch['desired_speed']  # 보통 [B,1] (장담 못하므로 아래에서 표준화)
        if isinstance(gt_tensor, torch.Tensor):
            gt_np = gt_tensor.detach().cpu().numpy()
        else:
            gt_np = gt_tensor
        gt_np = convert_11(gt_np)  # [-1,1] 범위
        ground_truth = torch.as_tensor(gt_np, dtype=torch.float32, device=self.device).view(-1, 1)  # [B,1]

        # ground_truth 값을 SafeTruncatedNormal 분포의 유효범위인 [-1, 1] 안으로 강제
        # 1e-6 (epsilon)을 추가하여 경계 값으로 인한 오류를 방지
        ground_truth = torch.clamp(ground_truth, -1.0 + 1e-6, 1.0 - 1e-6)

        # 4) 모든 speed head에 대해 IL 로스 누적
        total_imitation_loss = 0.0
        for i in range(self.n_speed_heads):
            # 3) 부트스트랩 마스크: ground_truth와 같은 모양/디바이스
            mask = torch.full_like(ground_truth, self.bootstrap_p, device=self.device)  # [B,1] / ground_truth 텐서와 동일한 shape & device를 가지는 새로운 텐서 mask 생성. 이는 self.bootstrap_p 값으로 다 채워짐
            mask = torch.bernoulli(mask) # [B,1] / 바로 위에서 생성한 mask 를 이용해 베르누이 분포에 따른 무작위 샘플링 수행
                                        # torch.bernoulli : 입력 텐서의 각 원소를 확률로 간주하여, 해당 확률에 따라 0 또는 1 무작위 생성
                                        # e.g. mask 텐서의 특정 위치의 값이 0.7이면, 그 위치의 결과는 70% 확률로 1이 되고, 30% 확률로 0이 됨
                                        # 즉, mask 텐서는 0 또는 1로만 binary mask로 변환됨. 즉, 학습에 포함시킬 샘플(1)과 포함시키지 않을 샘플(0)을 무작위로 결정
            if mask.sum() < 1:  # 안전장치(모두 0이면 학습 불가)
                mask[0, 0] = 1.0 # 강제로 amsk의 첫 번째 원소 [0, 0]을 1로 설정하여 최소 1개 데이터는 학습에 사용되도록
            
            dist = self.policy_speed[i].get_dist(state_batch)  # TruncatedNormal, Independent(event_dim=1)
            # Independent(…,1)이면 log_prob 결과는 보통 [B] → [B,1]로 맞춰서 mask와 동일 차원 사용
            logp = dist.log_prob(ground_truth).view(-1, 1)     # [B,1]
            loss = -(logp * mask).sum() / mask.sum()           # 스칼라
            total_imitation_loss = total_imitation_loss + loss

        # 5) 옵티마이저 zero_grad
        for i in range(self.n_speed_heads):
            self.policy_speed[i].optimizer.zero_grad(set_to_none=True)
        self.image_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoints_encoder.optimizer.zero_grad(set_to_none=True)
        self.vm_encoder.optimizer.zero_grad(set_to_none=True)

        # 6) 역전파
        total_imitation_loss.backward()

        # 7) 모든 speed head + 공유 인코더 업데이트
        for i in range(self.n_speed_heads):
            torch.nn.utils.clip_grad_norm_(self.policy_speed[i].parameters(), self.grad_clip)
            self.policy_speed[i].optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.waypoints_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.vm_encoder.parameters(), self.grad_clip)

        self.image_encoder.optimizer.step()
        self.waypoints_encoder.optimizer.step()
        self.vm_encoder.optimizer.step()

        # 8) 로깅
        metrics['policy_speed_loss'] = round(float(total_imitation_loss.item()), 4)
        return metrics
        

    @staticmethod
    def get_current_speed(obs):
        return obs['speed']['speed'][0]

    def reset(self, obs):
        self.pid.reset()
        self.img_deque = deque([], maxlen=self.deque_size)
        
        obs = self.filter_obs(obs)
        for i in range(self.deque_size):
            self.img_deque.append(obs['image'])

    def set_train_mode(self):
        self.critic.train()
        self.critic_target.train()
        for head in self.policy_speed:      ### 변경점 : 모든 speed head를 train 모드로
            head.train()
        self.policy_steer.train()
        self.image_encoder.train()
        self.waypoints_encoder.train()
        self.vm_encoder.train()

    def set_eval_mode(self):
        self.critic.eval()
        self.critic_target.eval()
        for head in self.policy_speed:     ### 변경점 : 모든 speed head를 eval 모드로
            head.eval()
        self.policy_steer.eval()
        self.image_encoder.eval()
        self.waypoints_encoder.eval()
        self.vm_encoder.eval()

    #### 변경점
    def save_models(self, save_memory=False):
        print(f'{Fore.GREEN} saving models... {Fore.RESET}')

        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

        # [FIX] 각 head 디렉토리 보장 생성 후 save
        for head in self.policy_speed:
            ckpt_dir = getattr(head, 'checkpoint_dir', None)
            if ckpt_dir is not None:
                os.makedirs(ckpt_dir, exist_ok=True)  # head_i
                os.makedirs(os.path.join(ckpt_dir, 'weights'), exist_ok=True)
                os.makedirs(os.path.join(ckpt_dir, 'weights', 'optimizers'), exist_ok=True)
            head.save_checkpoint()
        self.policy_steer.save_checkpoint()
        self.image_encoder.save_checkpoint()
        self.waypoints_encoder.save_checkpoint()
        self.vm_encoder.save_checkpoint()

        if self.automatic_entropy_tuning:
            os.makedirs(os.path.join(self.experiment_path, 'weights'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_path, 'weights', 'optimizers'), exist_ok=True)
            torch.save(self.log_alpha, f'{self.experiment_path}/weights/log_alpha.pt')
            torch.save(self.alpha_optim.state_dict(),
                    f'{self.experiment_path}/weights/optimizers/log_alpha.pt')

    def save_periodic_models(self, step):
        """Saves a periodic checkpoint of the models at a given step."""
        # 1. 새로운 저장 경로를 만들고, 폴더가 없으면 생성
        periodic_weights_dir = os.path.join(self.experiment_path, 'weights_10000')
        periodic_optimizers_dir = os.path.join(periodic_weights_dir, 'optimizers')
        os.makedirs(periodic_weights_dir, exist_ok=True)
        os.makedirs(periodic_optimizers_dir, exist_ok=True)
        
        print(f'{Fore.MAGENTA} saving periodic models at step {step}... {Fore.RESET}')

        # 2. 파일 이름에 붙일 접미사(suffix)를 정의. 예: _10000
        suffix = f"_{step}"

        # 3. 각 네트워크의 save_checkpoint 함수를 호출하며, 새로운 경로와 접미사를 전달합니다.
        #    (이를 위해 각 네트워크의 save_checkpoint 함수도 수정이 필요합니다. 아래 가이드 참고)
        self.critic.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)
        self.critic_target.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)
        
        for head in self.policy_speed:
            # Multi-head Speed Actor의 경우, 각 head의 checkpoint_dir을 기반으로 새로운 경로를 만듭니다.
            head_periodic_weights_dir = os.path.join(getattr(head, 'checkpoint_dir').replace(self.experiment_path, periodic_weights_dir))
            os.makedirs(os.path.join(head_periodic_weights_dir, 'optimizers'), exist_ok=True)
            head.save_checkpoint(directory=head_periodic_weights_dir, suffix=suffix)
            
        self.policy_steer.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)
        self.image_encoder.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)
        self.waypoints_encoder.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)
        self.vm_encoder.save_checkpoint(directory=periodic_weights_dir, suffix=suffix)

        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, f'{periodic_weights_dir}/log_alpha{suffix}.pt')
            torch.save(self.alpha_optim.state_dict(), f'{periodic_optimizers_dir}/log_alpha{suffix}.pt')

    def load_models(self, save_memory=False):
        print(f'{Fore.GREEN} loading models... {Fore.RESET}')

        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        for head in self.policy_speed:              ### 변경점 : 모든 speed actor head 로드
            head.load_checkpoint()
        self.policy_steer.load_checkpoint()
        self.image_encoder.load_checkpoint()
        self.waypoints_encoder.load_checkpoint()
        self.vm_encoder.load_checkpoint()

        
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(
                f'{self.experiment_path}/weights/log_alpha.pt', map_location=self.device)
            
            # alpha 옵티마이저 상태를 CPU로 먼저 불러옴
            alpha_optimizer_state = torch.load(f'{self.experiment_path}/weights/optimizers/log_alpha.pt', map_location='cpu')
            # 그 다음, 옵티마이저에 상태를 로드
            self.alpha_optim.load_state_dict(alpha_optimizer_state)
            
            self.alpha = self.log_alpha.exp()

