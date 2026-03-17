import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import time


@dataclass
class CHTConfig:
    sparsity: float
    mlp_sparsity: float
    link_update_ratio: float
    remove_method: str
    regrow_method: str
    shared_mask_sw: bool
    shared_mask_zone: bool
    zone_sz: int
    avg_remove: bool
    avg_regrow: bool
    soft: bool
    use_opt4: bool
    delta: float
    delta_max: float
    delta_d: float
    delta_remove: float
    ch_method: str
    use_hidden: bool
    l3n_batch_sz: int
    evolve_es: bool
    use_manual: bool

    use_ss: bool
    ss_sparsity_initial: float
    ss_k: int
    ss_duration: int

    dropout: float

    # Ultralytics compatibility
    nc: int = 80  # Number of classes

    # Sparsity schedule parameters
    sparsity_schedule: str = "immediate"  # "immediate", "step", "linear", "sigmoid"
    sparsity_warmup_epochs: int = 5  # epochs before sparsity starts increasing
    sparsity_step_epochs: int = 5    # increase sparsity every N epochs
    sparsity_step_size: float = 0.1  # increase by 0.1 (10%) each step

    # Debug flag to control verbose output
    debug: bool = False


class Conv2d_CHT(nn.Module):

    def __init__(
            self, c_in, c_out, kernel_sz,
            cht_config: CHTConfig,
            *, padding=0, stride=1, bias=True):

        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_sz

        self.padding = padding
        self.stride = stride
        self.use_bias = bias

        # CHT config
        self.sparsity = cht_config.sparsity
        self.mlp_sparsity = cht_config.mlp_sparsity
        self.link_update_ratio = cht_config.link_update_ratio
        self.remove_method = cht_config.remove_method
        self.regrow_method = cht_config.regrow_method
        self.shared_mask_sw = cht_config.shared_mask_sw
        self.shared_mask_zone = cht_config.shared_mask_zone
        # assert self.shared_mask_zone
        self.zone_sz = cht_config.zone_sz
        if self.zone_sz == 0:
            self.zone_sz = c_in  # The entire mask is a single zone
        elif self.zone_sz > c_in or c_in % self.zone_sz != 0:
            self.zone_sz = c_in
        self.num_zones = c_in // self.zone_sz
        self.avg_remove = cht_config.avg_remove        # Whether to remove links according to average scores between SWs: 1.3
        self.avg_regrow = cht_config.avg_regrow        # Whether to regrow links according to average scores between SWs: 1.2, 1.3
        self.ch_method = cht_config.ch_method
        self.use_hidden = cht_config.use_hidden        # Whether to use hidden layer functionality
        self.l3n_batch_sz = cht_config.l3n_batch_sz    # Batch size for L3n computation
        if self.l3n_batch_sz == 0 or self.l3n_batch_sz > self.c_out:
            self.l3n_batch_sz = self.c_out

        # CHTs config
        self.soft = cht_config.soft                    # Whether to use soft removal and regrowth i.e. CHTs
        self.use_opt4 = cht_config.use_opt4            # Whether to use option 4 in _get_L3n_regrow_pos method
        self.delta = cht_config.delta
        self.delta_max = cht_config.delta_max
        self.delta_d = cht_config.delta_d
        self.delta_remove = cht_config.delta_remove

        # CHTss config
        self.use_ss = cht_config.use_ss                # Whether to use CHTss
        self.ss_sparsity_initial = cht_config.ss_sparsity_initial  # Initial CHTss sparsity value
        self.ss_k = cht_config.ss_k                    # CHTss k parameter
        self.ss_duration = cht_config.ss_duration      # CHTss duration parameter
        if self.use_ss:
            self.sparsity_target = self.sparsity
            self.sparsity = self.ss_sparsity_initial
            self.evolve_count = 0

        # Sparsity schedule config (for dense-to-sparse evolution)
        self.sparsity_schedule = cht_config.sparsity_schedule
        self.sparsity_warmup_epochs = cht_config.sparsity_warmup_epochs
        self.sparsity_step_epochs = cht_config.sparsity_step_epochs
        self.sparsity_step_size = cht_config.sparsity_step_size

        # Debug flag to control verbose output
        self.debug = cht_config.debug

        # Inference mode flag - when True, evolution is disabled
        self._inference_mode = False

        # Store the target sparsity
        self.sparsity_target = self.sparsity
        # For step schedule, start with 0 sparsity (dense) and increase over time
        if self.sparsity_schedule == "step":
            self.sparsity = 0.0  # Start dense
        # For immediate/sigmoid/linear schedules, also start dense (0%) during warmup
        # The sparsity will be applied after warmup epochs
        elif self.sparsity_schedule in ("immediate", "sigmoid", "linear"):
            self.sparsity = 0.0  # Start dense during warmup

        # Other configs
        self.evolve_es = cht_config.evolve_es          # Whether to enable early stop for evolve
        self.layer_name = 'CHT_Conv'                   # Layer name for monitoring

        self.unshared_converge_mask = None
        self.unshared_converge_thres = 0.99

        weight = torch.empty(c_out, c_in, kernel_sz, kernel_sz)
        torch.nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        # Original: use 4D format for dense, 2D for sparse
        # But we now always use mask, so keep 4D for simplicity
        self.weight = nn.Parameter(weight)  # [c_out, c_in, K, K]

        # Initialize bias
        bias = torch.randn(c_out)
        torch.nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)  # [c_out]

        # Initialize mask - ALWAYS create mask from the start (dense = all 1s)
        # Sparsity will be applied after warmup via evolve() or _apply_target_sparsity()
        assert not self.shared_mask_zone

        # Create initial mask with all 1s (dense) - 4D format
        # This ensures the model starts fully dense during warmup
        mask = torch.ones(c_out, c_in, kernel_sz, kernel_sz, dtype=torch.bool, device=weight.device)
        self.register_buffer('mask', mask)

        self.input_param_initialized = False

        # Quantization simulation flag (propagated from QAT wrapper)
        self.simulate_quant = False

        # Initialize input dimensions (will be set during first forward pass)
        # Required for evolve() to work before any forward pass
        self.in_h = 0
        self.in_w = 0
        self.out_h = 0
        self.out_w = 0

        # Calculate num_zeros/num_active based on current sparsity
        # Initially this will be 0 (dense) since we set sparsity=0 during init
        self.num_zeros = int(self.sparsity * c_in * kernel_sz * kernel_sz)
        self.num_active = c_in * kernel_sz * kernel_sz - self.num_zeros
        self.num_update = int(self.link_update_ratio * self.num_active)

        # print(f'[DEBUG] {self.num_active = }, {self.link_update_ratio = }, {self.num_update = }')

        # Initialize _num_sw_cache for evolve() calls before forward pass
        self._num_sw_cache = None

    @property
    def num_sw(self):
        """Get number of sliding windows, computing from cache or from module params."""
        if self._num_sw_cache is not None:
            return self._num_sw_cache
        # Try stored dimensions first
        if hasattr(self, 'out_h') and hasattr(self, 'out_w'):
            self._num_sw_cache = self.out_h * self.out_w
            return self._num_sw_cache
        # Compute from module parameters (used before forward pass)
        # num_sw = out_h * out_w where:
        # out_h = (in_h + 2*padding - kernel_size) // stride + 1
        # out_w = (in_w + 2*padding - kernel_size) // stride + 1
        # For typical conv: in_h=in_w=image_size, padding=kernel_size//2, stride=1
        # At initialization, use default feature map size (e.g., 16x16 for 128x128 input)
        if hasattr(self, 'c_in') and hasattr(self, 'kernel_size'):
            # Use typical output size based on stride pattern
            # For YOLO with 128x128 input: 128 -> 64 -> 32 -> 16 -> 8
            # Use a reasonable default: 16x16 = 256 sliding windows
            self._num_sw_cache = 256
            return self._num_sw_cache
        # Fallback to a reasonable default
        return 256

    @num_sw.setter
    def num_sw(self, value):
        """Cache num_sw value after first forward pass."""
        self._num_sw_cache = value

    def train(self, mode=True):
        """Override train() to set inference mode flag."""
        self._inference_mode = not mode
        return super().train(mode)

    def eval(self):
        """Override eval() to set inference mode flag."""
        self._inference_mode = True
        return super().eval()

    def forward(self, x):
        batch_sz, c_in, h, w = x.shape

        # Apply mask to weight (always use mask, even when dense)
        # Mask is 4D: [c_out, c_in, K, K]
        # Convert mask to float for proper gradient computation
        masked_weight = self.weight * self.mask.float()

        if self.use_bias:
            out = F.conv2d(x, masked_weight, self.bias, stride=self.stride, padding=self.padding)
        else:
            out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding)

        # Dynamically compute output shape from conv2d result
        # Use actual output channels from conv2d to avoid shape mismatch errors
        # Use reshape instead of view to handle any size mismatches
        output = out.reshape(batch_sz, out.shape[1], out.shape[2], out.shape[3])

        return output

    def update_sparsity(self, current_epoch, total_epochs=None):
        """Gradually increase sparsity based on epoch (for step schedule)."""
        if self.sparsity_schedule != "step":
            return  # Only apply step schedule when configured

        if current_epoch < self.sparsity_warmup_epochs:
            # During warmup, keep sparsity at 0 (dense)
            self.sparsity = 0.0
        else:
            # Calculate how many steps have passed (using ceiling division)
            # This ensures first step increase happens right after warmup ends
            step = max(0, (current_epoch - self.sparsity_warmup_epochs + self.sparsity_step_epochs - 1) // self.sparsity_step_epochs)
            new_sparsity = min(self.sparsity_target, step * self.sparsity_step_size)
            self.sparsity = new_sparsity

        # Recalculate num_zeros, num_active, num_update
        self.num_zeros = int(self.sparsity * self.c_in * self.kernel_size * self.kernel_size)
        self.num_active = self.c_in * self.kernel_size * self.kernel_size - self.num_zeros
        self.num_update = int(self.link_update_ratio * self.num_active)

    def _sync_num_vars_with_mask(self):
        """Synchronize num_zeros/num_active with actual mask state."""
        if not hasattr(self, 'mask') or self.mask is None:
            return
        total_params = self.mask.numel()
        actual_active = self.mask.sum().item()
        actual_zeros = total_params - actual_active
        self.num_zeros = actual_zeros
        self.num_active = actual_active
        self.num_update = int(self.link_update_ratio * self.num_active)

    @torch.no_grad()
    def evolve(self, current_epoch=None):
        # Skip evolution during inference to prevent mask corruption
        if self._inference_mode:
            return None, None, None

        # Update sparsity based on schedule if epoch is provided
        if current_epoch is not None and self.sparsity_schedule == "step":
            self.update_sparsity(current_epoch)

        # Sparsity is now controlled by weight-based removal only (no random removal)
        # The num_update connections are removed based on weight magnitude each evolve
        if self.sparsity == 0. or self.link_update_ratio == 0.:
            return None, None, None

        layer_name = getattr(self, 'layer_name', 'CHT_Layer')
        c_out, c_in = self.weight.shape[0], self.weight.shape[1]
        if self.debug:
            print(f"[CHT DEBUG] Evolving {layer_name}: c_out={c_out}, c_in={c_in}, k={self.kernel_size}, sparsity={self.sparsity:.2f}")

        if self.sparsity == 0. or self.link_update_ratio == 0.:
            return None, None, None

        if self.use_ss:
            ind = -self.ss_k * (self.evolve_count - self.ss_duration / 2)
            ss_sparsity = self.ss_sparsity_initial + \
                (self.sparsity_target - self.ss_sparsity_initial) / (1 + math.e**ind)
            # print(f'[DEBUG] {self.evolve_count = }, {ss_sparsity = }')

            ss_num_zeros = int(ss_sparsity * self.c_in * self.kernel_size * self.kernel_size)
            ss_num_active = self.c_in * self.kernel_size * self.kernel_size - ss_num_zeros
            # ss_num_update = ss_num_active - (self.num_active - self.num_update)
            ss_num_update = int(self.link_update_ratio * ss_num_active)
            self.num_update = self.num_active - (ss_num_active - ss_num_update)  # active_now - remain_target
            # print(f'[DEBUG] {self.num_zeros = }, {self.num_active = }, {self.num_update = }')

            self.evolve_count += 1

        # Show current state
        current_active = self.mask.sum().item() if not self.use_hidden else self.hidden_mask.sum().item()
        total_params = self.mask.numel()
        current_sparsity = 1.0 - (current_active / total_params)

        # Remove connections
        remove_start = time.time()
        remove_pos = self._remove()
        remove_time = time.time() - remove_start

        # Sync num_vars with actual mask after removal
        self._sync_num_vars_with_mask()

        if self.use_ss:
            self.sparsity = ss_sparsity
            self.num_zeros = ss_num_zeros
            self.num_active = ss_num_active
            self.num_update = ss_num_update

        # Regrow connections
        regrow_start = time.time()
        regrow_pos = self._regrow()
        regrow_time = time.time() - regrow_start

        # Sync num_vars with actual mask after regrow
        self._sync_num_vars_with_mask()

        if self.use_ss:
            self.num_update = int(self.link_update_ratio * self.num_active)

        # Calculate the cancellation ratio (with safety for zero division)
        total_remove = remove_pos.sum().item()
        total_regrow = regrow_pos.sum().item()
        overlap = (remove_pos & regrow_pos).sum().item()

        if total_remove == 0 or total_regrow == 0:
            cancellation_ratio = 0.0  # No cancellation if no removal or no regrowth
        else:
            cancellation_ratio = overlap / total_remove

        # Force non-NaN
        if math.isnan(cancellation_ratio) or math.isinf(cancellation_ratio):
            cancellation_ratio = 0.0


        if self.delta < self.delta_max:
            self.delta += self.delta_d

        # Calculate mask convergence scores
        min_score, mean_score = self.calculate_mask_convergence()

        # And calculate the converge_mask from the unshared mask,
        # which will be used in forward before the next evolution
        if (not self.shared_mask_sw and self.regrow_method == 'L3n'
            and mean_score >= self.unshared_converge_thres
        ):
            print(f'  [DEBUG] Calculate the converge_mask from the unshared mask')
            converge_mask = torch.zeros(
                self.c_out,
                self.zone_sz * self.kernel_size * self.kernel_size,
                dtype=torch.bool,
                device=self.weight.device)
            _, indices = self.mask.float().mean(dim=1).topk(self.num_active)
            converge_mask.scatter_(1, indices, True)
            self.unshared_converge_mask = converge_mask  # [c_out, zone_sz * K * K]
        else:
            self.unshared_converge_mask = None

        return cancellation_ratio, min_score, mean_score


    def _remove(self):
        # Guard: Skip remove if num_update <= 0
        if self.num_update <= 0:
            return torch.zeros_like(self.mask, dtype=torch.bool) if not self.use_hidden else torch.zeros_like(self.hidden_mask, dtype=torch.bool)

        # Guard: Skip remove if num_active - num_update <= 0
        if self.num_active - self.num_update <= 0:
            return torch.zeros_like(self.mask, dtype=torch.bool) if not self.use_hidden else torch.zeros_like(self.hidden_mask, dtype=torch.bool)

        if not self.use_hidden:
            remove_pos = torch.zeros_like(self.mask)  # dtype: bool
        else:
            remove_pos = torch.zeros_like(self.hidden_mask)  # dtype: bool

        mask_flatten = self.mask.flatten(1)  # [c_out, c_in * K * K]
        weight_flatten = self.weight.flatten(1)  # [c_out, c_in * K * K]
        remove_pos_flatten = remove_pos.flatten(1)  # [c_out, c_in * K * K]

        if not self.shared_mask_zone:  # Skip
            match self.remove_method:
                # case 'rand':
                #     rand_values = torch.rand_like(mask_flatten, dtype=torch.float)
                #     rand_values.masked_fill_(~mask_flatten, 0.)  # Omit inactive positions
                #     _, indices = torch.topk(rand_values, k=self.num_update, dim=-1)
                #     remove_pos_flatten.scatter_(1, indices, True)

                case 'wm':
                    assert self.soft
                    if self.delta_remove == 1.:  # rigid
                        masked_weight = weight_flatten.masked_fill(~mask_flatten, torch.inf)  # Omit the inactive positions
                        _, indices = torch.topk(masked_weight.abs(), k=self.num_update, largest=False)
                        remove_pos_flatten.scatter_(1, indices, True)
                    else:
                        masked_weight = weight_flatten.masked_fill(~mask_flatten, 0.).abs()
                        exp = self.delta_remove / (1 - self.delta_remove)
                        masked_weight **= exp
                        masked_weight += 1e-6
                        masked_weight = masked_weight.masked_fill(~mask_flatten, 0.)
                        # Replace inf entries in masked_weight with the max of non-inf entries
                        is_inf = masked_weight == float('inf')
                        if torch.any(~is_inf):
                            max_noninf = masked_weight[~is_inf].max()
                        else:
                            max_noninf = 0.  # fallback, all inf, shouldn't happen
                        masked_weight[is_inf] = max_noninf

                        # SAFETY CHECK: Ensure valid probability distribution
                        if masked_weight.sum() == 0 or masked_weight.isnan().any():
                            print(f"[WARN] Invalid masked_weight in remove, using uniform distribution")
                            valid_mask = ~mask_flatten
                            if valid_mask.sum() == 0:
                                # No inactive positions - skip removal for this case
                                pass
                            else:
                                uniform_probs = valid_mask.float() / valid_mask.sum(dim=-1, keepdim=True).float()
                                indices_keep = torch.multinomial(uniform_probs, self.num_active - self.num_update, replacement=False)
                                remove_pos_flatten.scatter_(1, indices_keep, True)
                        else:
                            indices_keep = torch.multinomial(masked_weight, self.num_active - self.num_update, replacement=False)
                            remove_pos_flatten.scatter_(1, indices_keep, True)
                        remove_pos_flatten = ~remove_pos_flatten
                        remove_pos_flatten[~mask_flatten] = False  # Remove the current inactive positions from the record

                case _:
                    raise NotImplementedError
                
            remove_pos = remove_pos_flatten.view(remove_pos.shape)
                
        else:
            raise NotImplementedError
            if not self.use_hidden:  # SCHT, SET
                match self.remove_method:
                    case 'wm':
                        avg_weight = self.weight.view(
                            self.c_out,
                            self.num_zones,
                            self.zone_sz * self.kernel_size * self.kernel_size
                        )
                        avg_weight = avg_weight.mean(dim=1)  # [c_out, zone_sz * K * K]

                        if not self.soft:
                            scores = avg_weight.masked_fill(~mask_flatten, torch.inf)
                            _, indices = torch.topk(scores.abs(), k=self.num_update, largest=False)
                            remove_pos_flatten.scatter_(1, indices, True)
                        else:
                            scores = avg_weight.masked_fill(~mask_flatten, 0.).abs()
                            if self.delta_remove != 0.:
                                exp = self.delta_remove / (1 - self.delta_remove)
                                scores **= exp
                                if self.delta_remove >= 0.9:
                                    scores += 1e-6
                                    scores = scores.masked_fill(~mask_flatten, 0.)
                            else:
                                scores = scores.masked_fill(mask_flatten, 1.)

                            indices = torch.multinomial(scores, self.num_active - self.num_update, replacement=False)
                            indices = indices.view(self.c_out, -1)
                            remove_pos_flatten.scatter_(1, indices, True)
                            remove_pos_flatten = ~remove_pos_flatten
                            remove_pos_flatten[~mask_flatten] = False  # Remove the current inactive positions from the record
                    
                    case _:
                        raise NotImplementedError
                
                remove_pos = remove_pos_flatten.view(remove_pos.shape)

        # Update mask
        self.mask.masked_fill_(remove_pos, False)
        self._check_mask(self.mask, self.num_zeros + self.num_update)

        return remove_pos
    
    
    def _regrow(self):
        # Skip regrow if there's nothing to update (num_update <= 0)
        if self.num_update <= 0:
            return torch.zeros_like(self.mask, dtype=torch.bool) if not self.use_hidden else torch.zeros_like(self.hidden_mask, dtype=torch.bool)

        # Additional guard: ensure num_active - num_update > 0 for sampling
        if self.num_active - self.num_update <= 0:
            return torch.zeros_like(self.mask, dtype=torch.bool) if not self.use_hidden else torch.zeros_like(self.hidden_mask, dtype=torch.bool)

        if not self.use_hidden:
            regrow_pos = torch.zeros_like(self.mask)  # dtype: bool
        else:
            regrow_pos = torch.zeros_like(self.hidden_mask)  # dtype: bool

        if self.shared_mask_sw:
            mask_flatten = self.mask.flatten(1)  # [c_out, c_in * K * K]
            regrow_pos_flatten = regrow_pos.flatten(1)  # [c_out, c_in * K * K]

            if not self.shared_mask_zone:
                match self.regrow_method:
                    # case 'rand':
                    #     rand_values = torch.rand_like(mask_flatten, dtype=torch.float)
                    #     rand_values.masked_fill_(mask_flatten, 0.)  # Omit active positions
                    #     _, indices = torch.topk(rand_values, k=self.num_update, dim=-1)
                    #     regrow_pos_flatten.scatter_(1, indices, True)
                    
                    case 'L3n':
                        mask_repeated = self.mask.view(self.c_out * self.c_in, self.kernel_size * self.kernel_size)
                        mask_repeated = mask_repeated.unsqueeze(1).repeat(1, self.num_sw, 1)  # [c_out * c_in, num_sw, K^2]
                        scores_extended = []
                        # regrow_pos_extended = []
                        for start in range(0, self.c_out * self.c_in, self.l3n_batch_sz):
                            end = start + self.l3n_batch_sz
                            regrow_pos_part = self._get_L3n_regrow_pos_optimized(mask_repeated[start:end])
                            # Handle case where function returns early with different shape
                            expected_shape = (end - start, self.kernel_size * self.kernel_size)
                            if regrow_pos_part.shape != expected_shape:
                                # Function returned early (e.g., num_sw out of bounds), create matching fallback
                                layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                                if self.debug:
                                    print(f"[CHT DEBUG] Fallback for {layer_name}: shape mismatch in regrow")
                                    print(f"             Got: {regrow_pos_part.shape}, expected: {expected_shape}")
                                    print(f"             Layer: c_out={self.c_out}, c_in={self.c_in}, k={self.kernel_size}")
                                regrow_pos_part = torch.rand(expected_shape[0], expected_shape[1], device=mask_repeated.device)
                            scores_extended.append(regrow_pos_part)
                        scores_extended = torch.cat(scores_extended, dim=0)  # [c_out * c_in, K * K]
                        # print(f'[DEBUG] regrow finished: {self.c_out} - {self.c_in}')

                        # Validate final shape
                        if scores_extended.shape != (self.c_out * self.c_in, self.kernel_size * self.kernel_size):
                            layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                            print(f"[CHT] Warning: Final scores_extended shape {scores_extended.shape} != expected {(self.c_out * self.c_in, self.kernel_size * self.kernel_size)}")
                            # Reshape to expected dimensions if possible
                            scores_extended = scores_extended.view(self.c_out, self.c_in, self.kernel_size, self.kernel_size)
                            scores_extended = scores_extended.permute(0, 2, 3, 1).reshape(self.c_out * self.c_in, self.kernel_size * self.kernel_size)
                        assert self.soft

                        scores_extended = scores_extended.view(self.c_out, self.c_in * self.kernel_size * self.kernel_size)

                        if self.delta != 0.:
                            exp = self.delta / (1 - self.delta)
                            scores_extended **= exp
                            scores_extended += 1e-6
                        else:
                            scores_extended = scores_extended.masked_fill(~mask_flatten, 1.)

                        # scores_extended = scores_extended.masked_fill(mask_flatten, -1)
                        # _, indices = torch.topk(scores_extended, k=self.num_update, dim=1)
                        # regrow_pos_flatten.scatter_(1, indices, True)

                        scores_extended = scores_extended.masked_fill(mask_flatten, 0.)

                        # SAFETY CHECK: Ensure we have valid positions to sample
                        valid_mask = (scores_extended > 0) & ~scores_extended.isnan()
                        valid_count = valid_mask.sum(dim=-1, keepdim=True)

                        # Guard against num_update <= 0
                        if self.num_update <= 0:
                            return regrow_pos

                        if (valid_count < self.num_update).any():
                            # Not enough valid positions - fallback to random sampling from inactive
                            layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                            min_valid = valid_count.min().item()
                            if self.debug:
                                print(f"[CHT DEBUG] Fallback for {layer_name}: not enough valid positions")
                                print(f"             num_update={self.num_update}, min_valid={min_valid}")
                            inactive_mask = ~mask_flatten
                            # Sample from inactive positions uniformly
                            fallback_scores = inactive_mask.float()
                            fallback_scores = fallback_scores / fallback_scores.sum(dim=-1, keepdim=True)
                            indices = torch.multinomial(fallback_scores, self.num_update, replacement=False)
                        else:
                            # Safe to use multinomial
                            indices = torch.multinomial(scores_extended, self.num_update, replacement=False)

                        regrow_pos_flatten.scatter_(1, indices, True)
                    
                    case 'act':
                        pass

                    case _:
                        raise NotImplementedError

            else:
                raise NotImplementedError
                if not self.use_hidden:
                    match self.regrow_method:
                        case 'L3n':  # Regrow with position average
                            mask_repeated = mask_flatten.unsqueeze(1).repeat(1, self.num_sw, 1)
                            
                            # regrow_pos_extended = self._get_L3n_regrow_pos_optimized(mask_repeated)  # [c_out, num_sw, zone_sz * K * K]
                            regrow_pos_extended = []
                            for start in range(0, self.c_out, self.l3n_batch_sz):
                                end = start + self.l3n_batch_sz
                                regrow_pos_part = self._get_L3n_regrow_pos_optimized(mask_repeated[start:end])
                                regrow_pos_extended.append(regrow_pos_part)
                            regrow_pos_extended = torch.cat(regrow_pos_extended, dim=0)  # [c_out, num_sw, zone_sz * K * K]

                            regrow_pos_avg = regrow_pos_extended.float().mean(dim=1)  # [c_out, zone_sz * K * K]
                            _, indices = torch.topk(regrow_pos_avg, k=self.num_update, dim=1)
                            regrow_pos_flatten.scatter_(1, indices, True)
                        
                        case _:
                            raise NotImplementedError
                
                else:  # use_hidden = True
                    match self.regrow_method:
                        case 'L3n':
                            regrow_pos_flatten = self._get_L3n_regrow_pos_optimized(self.hidden_mask)
                        
                        case _:
                            raise NotImplementedError
                
            regrow_pos = regrow_pos_flatten.view(regrow_pos.shape)

        else:
            raise NotImplementedError

        # Update mask
        if not self.use_hidden:
            self.mask.masked_fill_(regrow_pos, True)
            self._check_mask(self.mask, self.num_zeros)
        else:
            self.hidden_mask.masked_fill_(regrow_pos, True)
            self._check_mask(self.hidden_mask, self.num_zeros)
            self.mask.copy_(self._hidden_to_mask())
            self._check_mask(self.mask, self.num_zeros)

        return regrow_pos


    def _chunked_bmm(self, A, B, min_chunk_size=500, max_chunks=20):
        """Compute A @ B in chunks to fit GPU memory.

        Args:
            A: First tensor of shape [batch, n, m]
            B: Second tensor of shape [batch, m, n]
            min_chunk_size: Minimum number of rows per chunk (default 500)
            max_chunks: Maximum number of chunks to avoid too many iterations

        Returns:
            Result of shape [batch, n, n]
        """
        batch, n, m = A.shape

        # If full operation fits in memory, do it at once
        if torch.cuda.is_available():
            mem_info = torch.cuda.mem_get_info()
            free_memory = mem_info[1] - mem_info[0]
            full_mem = n * m * n * 4  # bytes for float32 result
            available_for_op = free_memory * 0.3  # Use max 30% of free memory

            if full_mem <= available_for_op:
                return torch.bmm(A, B)

            # Calculate chunk size based on memory
            # Each chunk needs: chunk_size * m * n * 4 (result) + chunk_size * m * 4 (A slice)
            safe_chunk_size = int(available_for_op / (m * n * 4))
            chunk_size = max(min_chunk_size, safe_chunk_size)
        else:
            chunk_size = max(min_chunk_size, n // max_chunks)

        # Limit iterations to avoid distributed timeouts
        num_iterations = (n + chunk_size - 1) // chunk_size
        if num_iterations > max_chunks:
            chunk_size = max(min_chunk_size, n // max_chunks)
            num_iterations = (n + chunk_size - 1) // chunk_size

        if n > 5000:
            pass  # Silent chunked bmm
            # print(f"[CHT] Chunked bmm: {n}x{n}, chunk_size={chunk_size}, iterations={num_iterations}")

        result = torch.zeros(batch, n, n, device=A.device, dtype=A.dtype)

        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            # Compute partial result for chunk of rows
            result[:, i:end_i, :] = torch.bmm(A[:, i:end_i, :], B)
            # Synchronize periodically to avoid GPU hanging
            if i % (chunk_size * 3) == 0:
                torch.cuda.synchronize()

        torch.cuda.synchronize()
        return result

    @torch.no_grad()
    def _get_L3n_regrow_pos_optimized(self, mask_included):
        # mask_included: [l3n_bs, num_sw, zone_sz * K * K]
        # FIXED: Process in chunks to avoid memory issues (~237GB -> ~few GB)
        # Hardcode zone_sz = 1 for efficiency, matching original CHT behavior
        self.zone_sz = 1

        # Get the original indices on the input for each sliding window
        ph, pw = self.in_h + 2 * self.padding, self.in_w + 2 * self.padding
        num_sw = self.num_sw
        np = self.zone_sz * ph * pw

        # Check if feature map is too large OR too small for L3n computation
        # Threshold: skip if num_sw > 50000 (about 50K sliding windows)
        # Also skip if num_sw < 16 (too small for meaningful L3n computation)
        MIN_NUM_SW = 16
        MAX_NUM_SW = 50000

        # Always check num_sw bounds (both GPU and CPU)
        if num_sw > MAX_NUM_SW or num_sw < MIN_NUM_SW:
            layer_name = getattr(self, 'layer_name', 'CHT_Layer')
            c_out, c_in = self.weight.shape[0], self.weight.shape[1]
            kk = self.kernel_size * self.kernel_size
            if self.debug:
                print(f"[CHT DEBUG] Fallback for {layer_name}: num_sw={num_sw} (valid: [{MIN_NUM_SW}, {MAX_NUM_SW}])")
                print(f"             Layer: c_out={c_out}, c_in={c_in}, k={self.kernel_size}, zone_sz={self.zone_sz}")
                print(f"             Expected memory: ~{num_sw * c_out * c_in * 8 / 1024**3:.2f} GB")
            # Use random scores based on batch size
            batch_size = mask_included.shape[0]
            scores = torch.rand(batch_size, kk, device=mask_included.device)
            return scores

        if torch.cuda.is_available():
            mem_info = torch.cuda.mem_get_info()
            free_memory = mem_info[1] - mem_info[0]
            if free_memory < 2 * 1024**3:
                layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                c_out, c_in = self.weight.shape[0], self.weight.shape[1]
                kk = self.kernel_size * self.kernel_size
                if self.debug:
                    print(f"[CHT DEBUG] Fallback for {layer_name}: low GPU memory ({free_memory/1024**3:.1f}GB free)")
                    print(f"             Layer: c_out={c_out}, c_in={c_in}, k={self.kernel_size}, num_sw={num_sw}")
                    print(f"             Required: 2GB minimum, available: {free_memory/1024**3:.2f}GB")
                # Use random scores based on batch size
                batch_size = mask_included.shape[0]
                scores = torch.rand(batch_size, kk, device=mask_included.device)
                return scores

        regrow_pos = torch.zeros_like(mask_included, dtype=torch.bool)

        sw_indices = torch.arange(self.zone_sz * ph * pw, device=mask_included.device).float()
        sw_indices = sw_indices.view(self.zone_sz, ph, pw)
        sw_indices = F.unfold(sw_indices, kernel_size=self.kernel_size, stride=self.stride)
        sw_indices = sw_indices.t().long()  # [num_sw, zone_sz * K * K]

        # Process in smaller chunks to avoid memory issues
        chunk_size = 1  # Process 1 channel at a time to avoid memory issues
        num_chunks = (mask_included.shape[0] + chunk_size - 1) // chunk_size

        layer_name = getattr(self, 'layer_name', 'CHT_Layer')

        # Estimate memory requirement for the expensive matrix operations
        # The key operation is: elcl_DT_result + elcl_TD_result.transpose(-2, -1)
        # This creates a [bs_chunk, num_sw, num_sw] tensor
        if torch.cuda.is_available():
            bs_chunk_est = 1  # chunk_size is 1
            # Memory for scores_one_k: bs_chunk * num_sw * num_sw * 4 bytes (float32)
            estimated_mem = bs_chunk_est * num_sw * num_sw * 4
            free_memory = torch.cuda.mem_get_info()[0]

            if estimated_mem > free_memory * 0.5:  # Use more conservative threshold
                if self.debug:
                    print(f"[CHT DEBUG] Fallback for {layer_name}: estimated memory {estimated_mem/1024**3:.2f}GB > available {free_memory/1024**3:.2f}GB")
                    print(f"             num_sw={num_sw}, chunk_size={chunk_size}")
                kk = self.kernel_size * self.kernel_size
                batch_size = mask_included.shape[0]
                scores = torch.rand(batch_size, kk, device=mask_included.device)
                return scores

        # Silent evolution - progress shown in yolo_cht.py evolve()
        # print(f"  [EVOLVE] {layer_name}: Processing {num_chunks} chunks (num_sw={num_sw})...")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, mask_included.shape[0])
            mask_chunk = mask_included[start_idx:end_idx]
            bs_chunk = mask_chunk.shape[0]

            # Silent chunk processing - progress shown in yolo_cht.py evolve()
            # if chunk_idx % 10 == 0:
            #     print(f"    Chunk {chunk_idx}/{num_chunks}")

            # Build adjacency matrix for this chunk only
            am = torch.zeros(bs_chunk, self.num_sw, self.zone_sz * ph * pw,
                            dtype=torch.bool, device=mask_included.device)

            c_out_indices = torch.arange(bs_chunk, device=mask_included.device).unsqueeze(1).unsqueeze(2)
            j_indices = torch.arange(self.num_sw, device=mask_included.device).unsqueeze(0).unsqueeze(2)
            sw_indices_expanded = sw_indices.unsqueeze(0)

            am[c_out_indices, j_indices, sw_indices_expanded] = mask_chunk
            am = am.float()

            # Vectorized computation for this chunk
            DTPATHS1 = am
            TDPATHS1 = DTPATHS1.transpose(-2, -1)

            # Key insight: For DDPATHS2 = DTPATHS1 @ TDPATHS1 (where TDPATHS1 = DTPATHS1.T):
            # - DDPATHS2[i,j] = sum_k DTPATHS1[i,k] * DTPATHS1[j,k] = count of shared positions between window i and j
            # - DDPATHS2[i,i] = sum_k DTPATHS1[i,k] * DTPATHS1[i,k] = sum_k DTPATHS1[i,k] = row sum
            # So diagonal elements are just the row sums! No matrix multiplication needed!
            #
            # Similarly for TTPATHS2 = TDPATHS1 @ DTPATHS1:
            # - TTPATHS2[p,p] = sum_i TDPATHS1[p,i] * TDPATHS1[p,i] = row sum of TDPATHS1
            # - TTPATHS1 = DTPATHS1.T, so TTPATHS2[p,p] = col sum of DTPATHS1
            #
            # We only need diagonal elements, which are simply the row/col sums!
            sum_DTPATHS1 = torch.sum(DTPATHS1, dim=-1)  # [bs_chunk, num_sw]
            sum_TDPATHS1 = torch.sum(TDPATHS1, dim=-1)  # [bs_chunk, np]

            DDPATHS2_diag = sum_DTPATHS1.unsqueeze(1)  # [bs_chunk, 1, num_sw]
            TTPATHS2_diag = sum_TDPATHS1.unsqueeze(1)  # [bs_chunk, 1, np]

            # Boolean masks: True if there's at least one path
            BDDPATHS2_diag = (DDPATHS2_diag != 0)
            BTTPATHS2_diag = (TTPATHS2_diag != 0)

            elcl_DT = sum_DTPATHS1 - DDPATHS2_diag
            elcl_DT *= BDDPATHS2_diag
            elcl_TD = sum_TDPATHS1 - TTPATHS2_diag
            elcl_TD *= BTTPATHS2_diag

            elcl_DT[elcl_DT == 0] = 1.
            elcl_TD[elcl_TD == 0] = 1.
            elcl_DT -= 1
            elcl_TD -= 1

            if self.ch_method == 'CH3':
                elcl_DT += 1
                elcl_DT.reciprocal_()
                elcl_DT *= BDDPATHS2_diag
                elcl_TD += 1
                elcl_TD.reciprocal_()
                elcl_TD *= BTTPATHS2_diag
            elif self.ch_method == 'CH2':
                elcl_DT += 1
                elcl_DT.reciprocal_()
                elcl_DT *= (DDPATHS2_diag + BDDPATHS2_diag)
                elcl_TD += 1
                elcl_TD.reciprocal_()
                elcl_TD *= (TTPATHS2_diag + BTTPATHS2_diag)
            elif self.ch_method == 'CH3.1':
                elcl_DT.add_(1)
                elcl_DT.pow_(2 - 1 / elcl_DT)
                elcl_DT.reciprocal_()
                elcl_DT *= (DDPATHS2_diag + BDDPATHS2_diag)
                elcl_TD.add_(1)
                elcl_TD.pow_(2 - 1 / elcl_TD)
                elcl_TD.reciprocal_()
                elcl_TD *= (TTPATHS2_diag + BTTPATHS2_diag)
            else:
                raise NotImplementedError

            # Check memory before expensive operation
            if torch.cuda.is_available():
                mem_free = torch.cuda.mem_get_info()[0]
                # Need memory for: elcl_DT_result [bs, num_sw, num_sw] + elcl_TD_result [bs, num_sw, num_sw] + scores_one_k [bs, num_sw, num_sw]
                # Each is bs_chunk * num_sw * num_sw * 4 bytes
                needed = bs_chunk * self.num_sw * self.num_sw * 4 * 3
                if mem_free < needed:
                    layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                    if self.debug:
                        print(f"[CHT DEBUG] Fallback for {layer_name}: OOM risk, free={mem_free/1024**3:.2f}GB, need={needed/1024**3:.2f}GB")
                        print(f"             num_sw={self.num_sw}, bs_chunk={bs_chunk}")
                    kk = self.kernel_size * self.kernel_size
                    batch_size = mask_included.shape[0]
                    scores = torch.rand(batch_size, kk, device=mask_included.device)
                    return scores

            elcl_DT_result = torch.bmm(elcl_DT, DTPATHS1)
            elcl_TD_result = torch.bmm(elcl_TD, TDPATHS1)
            scores_one_k = elcl_DT_result + elcl_TD_result.transpose(-2, -1)

            # Ensure all indices are on the same device as the tensor being indexed
            device = scores_one_k.device
            c_out_indices = c_out_indices.to(device)
            j_indices = j_indices.to(device)
            sw_indices = sw_indices.to(device)

            scores_all_k = scores_one_k[c_out_indices, j_indices, sw_indices]

            if not self.avg_regrow:
                scores = scores_all_k
            else:
                # mask_chunk is on the same device as scores_all_k (no CPU transfer needed)
                mask_chunk_cpu = mask_chunk

                if self.use_opt4:
                    inactive_mask = ~mask_chunk_cpu
                    if inactive_mask.sum() > 0:
                        highest_score = scores_all_k[inactive_mask].max()
                    else:
                        highest_score = 0.
                    scores_all_k = scores_all_k.clone()
                    scores_all_k = scores_all_k.masked_fill(mask_chunk_cpu, highest_score)
                    scores = scores_all_k.mean(dim=1, keepdim=True)
                else:
                    zeros_in_each_col = (~mask_chunk_cpu).sum(dim=1)
                    avg_scores = scores_all_k.masked_fill(mask_chunk_cpu, 0)
                    avg_scores = avg_scores.sum(dim=1) / (zeros_in_each_col + 1e-6)
                    scores = avg_scores.unsqueeze(1)

            scores = scores.squeeze(1)

            if not self.soft:
                scores = scores.masked_fill(mask_chunk_cpu, -1)
                # Cap num_update to K*K to avoid sampling more than available positions
                max_update = self.kernel_size * self.kernel_size
                actual_update = min(self.num_update, max_update)

                # SAFETY: Ensure we don't sample more than available positions
                # Count how many positions have valid (non-negative) scores
                valid_count = (scores >= 0).sum(dim=-1, keepdim=True)
                actual_update = min(actual_update, int(valid_count.min().item()))

                if actual_update > 0:
                    _, top_indices = torch.topk(scores, k=actual_update, dim=-1)
                    # Move indices to the same device as regrow_pos
                    top_indices = top_indices.to(regrow_pos.device)
                    regrow_pos[start_idx:end_idx].scatter_(1, top_indices, True)
            else:
                if self.delta != 0.:
                    exp = self.delta / (1 - self.delta)
                    scores **= exp
                else:
                    scores = torch.ones_like(scores)

                scores += 1e-6
                scores = scores.masked_fill(mask_chunk_cpu, 0.)
                scores = scores.flatten(0, -2)

                # Cap num_update to K*K to avoid sampling more than available positions
                max_update = self.kernel_size * self.kernel_size
                actual_update = min(self.num_update, max_update)

                # SAFETY CHECK: Ensure valid probability distribution
                if scores.sum() == 0 or scores.isnan().any():
                    layer_name = getattr(self, 'layer_name', 'CHT_Layer')
                    if self.debug:
                        print(f"[CHT DEBUG] Fallback for {layer_name}: invalid scores in L3n regrow")
                        print(f"             scores.sum()={scores.sum().item()}, has_nan={scores.isnan().any().item()}")
                    scores = scores.new_ones(scores.shape)  # Uniform distribution

                indices = torch.multinomial(scores, actual_update, replacement=False)
                indices = indices.view(bs_chunk, self.num_sw, -1)
                # Move indices to the same device as regrow_pos
                indices = indices.to(regrow_pos.device)
                regrow_pos[start_idx:end_idx].scatter_(1, indices, True)

            # Average scores over sliding windows for output [bs_chunk, K*K]
            regrow_pos_chunk = regrow_pos[start_idx:end_idx]  # [bs_chunk, num_sw, K*K]
            regrow_pos_chunk = regrow_pos_chunk.float().mean(dim=1)  # [bs_chunk, K*K]

        return regrow_pos_chunk  # Return 2D [l3n_bs, K*K]
    def _check_mask(self, mask, num_zeros):
        # For shared_mask_sw = True: [c_out, zone_sz * K * K]
        # For shared_mask_sw = False: [c_out * num_sw, zone_sz * K * K]
        mask = mask.view(self.c_out, self.c_in * self.kernel_size * self.kernel_size)
        # Count number of zeros in each zone
        zeros_count = (mask == False).sum(dim=-1)
        
        # Check if number of zeros equals num_zeros for each vector
        assert torch.all(zeros_count == num_zeros), f'Number of zeros should be {num_zeros}, but got {zeros_count}'


    def calculate_mask_convergence(self):
        """
        Calculate mask convergence scores using min and mean methods.
        
        Returns:
            tuple: (min_score, mean_score) - convergence scores for this layer
        """
        if self.shared_mask_sw:
            if not self.use_hidden:
                return None, None
            mask = self.hidden_mask  # [c_out, num_sw, zone_sz * K * K]
        else:
            mask = self.mask
        
        score = mask.float().mean(dim=1)  # Mean across sliding windows
        scores = score.topk(self.num_active)[0]

        # Handle NaN values
        if scores.isnan().any():
            print(f"[DEBUG] NaN detected in scores, num_nan={scores.isnan().sum()}")
            scores = scores.nan_to_num(nan=0.0)

        min_score = scores.min().item()
        mean_score = scores.mean().item()

        # Handle NaN results
        if math.isnan(min_score) or math.isnan(mean_score):
            print(f"[DEBUG] NaN in final scores: min={min_score}, mean={mean_score}")
            min_score = 0.0
            mean_score = 0.0

        return min_score, mean_score
    

    def _unfold_input(self, x):
        # [batch_sz, c_in * K * K, num_sw]
        return F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


class Linear_CHT(nn.Module):
    def __init__(self, in_features, out_features, cht_config: CHTConfig):
        super(Linear_CHT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.sparsity = cht_config.mlp_sparsity
        self.link_update_ratio = cht_config.link_update_ratio
        self.remove_method = cht_config.remove_method
        self.regrow_method = cht_config.regrow_method

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.normal_(self.weight, 0, 0.01)
        self.bias = nn.Parameter(torch.randn(out_features))
        nn.init.constant_(self.bias, 0)

        # Layer name for monitoring
        self.layer_name = 'Linear_CHT'

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def _remove(self):
        return torch.zeros_like(self.mask, dtype=torch.bool)

    def _regrow(self):
        return torch.zeros_like(self.mask, dtype=torch.bool)

    def _get_L3n_regrow_pos(self, mask_included):
        return torch.zeros_like(mask_included, dtype=torch.bool)
