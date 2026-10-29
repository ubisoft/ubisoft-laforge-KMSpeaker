# Standard Library
import logging
import pickle

# External Modules
import torch
from torch.optim.lr_scheduler import LRScheduler

# Internal Modules
from kmspeaker.utils.flow_matching_utils import CondOTProbPath

logger = logging.getLogger(__name__)


class FlowTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        device,
        training_target: str,
        cfg_audio_keypoint_disentanglement=None,
        data_root=None,
        clip_grad_norm: float = None,
        loss_weights: dict = None,
        use_mesh_loss: bool = False,
        pca_decode_metadata: dict = None,
        keypoint_style_encoder: torch.nn.Module = None,
        cfg_probability: float = 0.0,
        target_codes="pca",
    ):
        """cfg_probability is the classifier-free guidance dropout rate; target_codes only supports 'pca'."""
        if training_target not in ("velocity_field", "x1_predictor"):
            raise ValueError("training_target must be either 'velocity_field' or 'x1_predictor'")
        if target_codes != "pca":
            raise ValueError(f"Invalid target_codes {target_codes}, only 'pca' is supported.")
        self.predict_x1 = training_target == "x1_predictor"
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.path = CondOTProbPath()
        self.loss_weights = loss_weights
        self.clip_grad_norm = clip_grad_norm
        self.use_mesh_loss = use_mesh_loss
        self.pca_decode_metadata = pca_decode_metadata
        self.data_root = data_root
        self.keypoint_style_encoder = keypoint_style_encoder  # Initialize keypoint style encoder
        self.use_keypoint_style_encoder = keypoint_style_encoder is not None
        self.prep_audio_keypoint_disentanglement_metadata(cfg_audio_keypoint_disentanglement)
        self.disentanglement = cfg_audio_keypoint_disentanglement is not None
        self.cfg_probability = cfg_probability
        self.target_codes = target_codes

        if not self.predict_x1 and "velocity_loss" not in self.loss_weights:
            raise ValueError("velocity loss can only be used if predicting x1")
        if self.loss_weights.get("vertex_mse_loss", 0) > 0 and not self.use_mesh_loss:
            raise ValueError(f"vertex_mse_loss != 0 (={self.loss_weights['vertex_mse_loss']}) but use_mesh_loss is False. " "Please set use_mesh_loss to True to compute vertex_mse_loss.")
        if self.disentanglement:
            if not self.use_mesh_loss:
                raise ValueError("Mesh loss should be enabled for disentanglement.")
            if self.loss_weights.get("vertex_mse_loss", 0) <= 0:
                raise ValueError("vertex_mse_loss should be >0 for disentanglement.")

    def prep_audio_keypoint_disentanglement_metadata(self, cfg_audio_keypoint_disentanglement):
        if cfg_audio_keypoint_disentanglement is not None:
            with open(self.data_root / cfg_audio_keypoint_disentanglement.metadata, "rb") as f:
                metadata = pickle.load(f)
            mask_to_get_audio = cfg_audio_keypoint_disentanglement.audio_type
            mask_to_get_keypoints = cfg_audio_keypoint_disentanglement.keypoint_type

            logger.info(f"Using mask type: {mask_to_get_audio} for audio disentanglement. ")
            logger.info(f"Using mask type: {mask_to_get_keypoints} for keypoint disentanglement. ")
            masks_data_audio = metadata[mask_to_get_audio]["audio_mask"]
            masks_data_keypoint = metadata[mask_to_get_keypoints]["keypoint_mask"]

            self.audio_mask = torch.tensor(masks_data_audio, device=self.device, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            self.keypoint_mask = torch.tensor(masks_data_keypoint, device=self.device, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # --- Loss computation helper functions ---
    def _resolve_masked_weight(self, base_name, mask, mask_name):
        """Returns loss_weights[base_name], or the mask-specific `<base_name>_audio`/`_keypoints` weight if mask is set."""
        if mask is None:
            return self.loss_weights.get(base_name, 0)
        if mask_name == "audio_mask" and self.loss_weights.get(f"{base_name}_audio", 0) > 0:
            return self.loss_weights.get(f"{base_name}_audio", 0)
        if mask_name == "keypoint_mask" and self.loss_weights.get(f"{base_name}_keypoints", 0) > 0:
            return self.loss_weights.get(f"{base_name}_keypoints", 0)
        raise ValueError(
            f"No valid weight configured for '{base_name}' with mask_name={mask_name}. "
            f"{base_name}_audio={self.loss_weights.get(f'{base_name}_audio', 0)}, "
            f"{base_name}_keypoints={self.loss_weights.get(f'{base_name}_keypoints', 0)}"
        )

    def add_flow_loss(self, pred, target, loss_dict):
        weight = self.loss_weights.get("flow_matching_loss", 0)
        if weight > 0:
            loss = torch.pow(pred - target, 2).mean()
            loss_dict["flow_matching_loss"] = loss.item()
            return weight * loss
        return 0.0

    def add_velocity_loss(self, pred, target, loss_dict, mask=None, mask_name=None):
        weight = self._resolve_masked_weight("velocity_loss", mask, mask_name)
        if weight > 0:
            gt_velocity = target[:, 1:] - target[:, :-1]
            pred_velocity = pred[:, 1:] - pred[:, :-1]
            if mask is not None:
                loss = (mask * torch.pow(pred_velocity - gt_velocity, 2)).mean()
            else:
                loss = torch.pow(pred_velocity - gt_velocity, 2).mean()

            key = f'velocity_loss{("_" + mask_name) if mask_name else ""}'
            loss_dict[key] = loss.item()
            return weight * loss
        return 0.0

    def add_smooth_loss(self, pred, loss_dict, mask=None, mask_name=None):
        weight = self.loss_weights.get("smooth_loss", 0)
        if weight > 0:
            diff = pred[:, 1:] - pred[:, :-1]
            if mask is not None:
                loss = torch.mean(mask * diff**2)
            else:
                loss = torch.mean(diff**2)
            key = f'smooth_loss{("_" + mask_name) if mask_name else ""}'
            loss_dict[key] = loss.item()
            return weight * loss
        return 0.0

    def add_vertex_mse_loss(self, pred, target, loss_dict, mask=None, mask_name=None):
        weight = self._resolve_masked_weight("vertex_mse_loss", mask, mask_name)
        if weight > 0:
            if mask is not None:
                loss = (mask * torch.pow(pred - target, 2)).mean()  # Ensures mask is of correct size (in case of no teeth)
            else:
                loss = torch.pow(pred - target, 2).mean()
            key = f'vertex_mse_loss{("_" + mask_name) if mask_name else ""}'
            loss_dict[key] = loss.item()
            return weight * loss
        return 0.0

    def add_kl_div_loss(self, mu_logvar, loss_dict):
        mu, logvar = mu_logvar
        assert mu is not None and logvar is not None, "mu and logvar are None - They must be provided for KL divergence loss."
        weight = self.loss_weights.get("kl_div_loss", 0)
        if weight > 0:
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            loss_dict["kl_div_loss"] = kl_div_loss.item()
            return weight * kl_div_loss
        return 0.0

    def _decode_mesh_from_codes(self, pred_codes, template):
        """Decodes predicted PCA codes back into per-vertex mesh coordinates."""
        if self.target_codes != "pca":
            raise ValueError(f"Invalid target_codes {self.target_codes} for mesh decoding, only 'pca' is supported.")
        inverse_transformed = torch.matmul(pred_codes, self.pca_decode_metadata["pca_components"]) + self.pca_decode_metadata["pca_mean"]
        V = self.pca_decode_metadata["pca_components"].shape[1] // 3
        pred_mesh = inverse_transformed.reshape(pred_codes.shape[0], -1, V, 3)
        if template is not None and len(template.shape) != 1:
            # pred_mesh is [batch_size, seq_len, n_vertices, 3], template is [batch_size, n_vertices, 3]
            pred_mesh += template.unsqueeze(1).to(self.device)
        return pred_mesh

    def _compute_disentanglement_loss(self, pred_mesh, target_mesh, disentanglement_key, loss_dict):
        """
        Computes masked vertex/velocity/smooth losses per disentanglement group: the
        audio-mask region for key==1 samples (audio right, keypoints wrong) and the
        keypoint-mask region for key==2 samples (keypoints right, audio wrong).
        """
        loss = 0.0
        for key_val, mask_name, mask in [
            (1, "audio_mask", self.audio_mask),
            (2, "keypoint_mask", self.keypoint_mask),
        ]:
            idx = (disentanglement_key == key_val).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            pred_sel = pred_mesh[idx]
            target_sel = target_mesh[idx]
            mask_tensor = mask[:, :, : pred_sel.shape[2]]
            loss += self.add_vertex_mse_loss(pred_sel, target_sel, loss_dict, mask=mask_tensor, mask_name=mask_name)
            if self.predict_x1:
                loss += self.add_velocity_loss(pred_sel, target_sel, loss_dict, mask=mask_tensor, mask_name=mask_name)
                loss += self.add_smooth_loss(pred_sel, loss_dict, mask=mask_tensor, mask_name=mask_name)

        # Track totals across groups separately, alongside the per-group breakdown
        loss_dict["vertex_mse_loss"] = sum(v for k, v in loss_dict.items() if k.startswith("vertex_mse_loss"))
        loss_dict["velocity_loss"] = sum(v for k, v in loss_dict.items() if k.startswith("velocity_loss"))
        loss_dict["smooth_loss"] = sum(v for k, v in loss_dict.items() if k.startswith("smooth_loss"))
        return loss

    def _compute_loss(self, batch, pred, dx_t, disentanglement_key=None, template=None, mu_logvar=(None, None)):
        """Computes the total training loss and per-term loss dictionary."""
        loss_dict = {}
        x_1 = batch["animation"].to(self.device)
        x_1_mesh = batch["mesh"].to(self.device) if "mesh" in batch else None

        pred_codes = pred

        if self.predict_x1:
            target = x_1
        else:
            target = dx_t

        target_codes = target

        target_mesh, pred_mesh = None, None
        if self.use_mesh_loss:
            target_mesh = x_1_mesh
            pred_mesh = self._decode_mesh_from_codes(pred_codes, template)

        if self.disentanglement and disentanglement_key is not None:
            loss = self._compute_disentanglement_loss(pred_mesh, target_mesh, disentanglement_key, loss_dict)
            if self.use_keypoint_style_encoder:
                loss += self.add_kl_div_loss(mu_logvar, loss_dict)

        else:

            loss = self.add_flow_loss(pred_codes, target_codes, loss_dict)
            if self.use_mesh_loss:
                loss += self.add_vertex_mse_loss(pred_mesh, x_1_mesh, loss_dict)
                if self.predict_x1:
                    loss += self.add_velocity_loss(pred_mesh, x_1_mesh, loss_dict)
                    loss += self.add_smooth_loss(pred_mesh, loss_dict)
            else:
                if self.predict_x1:
                    loss += self.add_velocity_loss(pred_codes, target_codes, loss_dict)
                    loss += self.add_smooth_loss(pred_codes, loss_dict)

        return loss, loss_dict

    def _apply_cfg_dropout(self, keypoint):
        """Randomly zeroes out keypoint conditioning (classifier-free guidance dropout)."""
        if self.cfg_probability > 0 and keypoint is not None:
            cfg_mask = torch.rand(keypoint.shape[0]) < self.cfg_probability
            keypoint = torch.where(cfg_mask[:, None, None].to(self.device), torch.zeros_like(keypoint), keypoint)
        return keypoint

    def _shuffle_for_disentanglement(self, audio, keypoint, disentanglement_key):
        """
        For key==1 samples, shuffles keypoints across the batch so audio stays aligned with
        the target but keypoints don't. For key==2 samples, shuffles audio instead, so
        keypoints stay aligned with the target but audio doesn't.
        """
        if disentanglement_key is None:
            return audio, keypoint

        if not torch.is_tensor(disentanglement_key):
            disentanglement_key = torch.tensor(disentanglement_key, device=self.device)
        audio = audio.clone()
        keypoint = keypoint.clone() if keypoint is not None else None

        idx_audio_right = (disentanglement_key == 1).nonzero(as_tuple=True)[0]
        if idx_audio_right.numel() > 0:
            perm = torch.randperm(idx_audio_right.numel())
            keypoint[idx_audio_right] = keypoint[idx_audio_right[perm]]

        idx_kpt_right = (disentanglement_key == 2).nonzero(as_tuple=True)[0]
        if idx_kpt_right.numel() > 0:
            perm = torch.randperm(idx_kpt_right.numel())
            audio[idx_kpt_right] = audio[idx_kpt_right[perm]]

        return audio, keypoint

    def train_step(self, batch: dict):
        """Runs one training step: forward pass, loss computation, backward pass, and optimizer/scheduler update."""
        self.model.train()
        self.optimizer.zero_grad()

        # --- Shared batch preparation ---
        x_1 = batch["animation"].to(self.device)
        audio = batch["audio"].to(self.device)
        keypoint = batch["keypoint"].to(self.device) if "keypoint" in batch else None
        extra_style_keypoint = batch["extra_style_keypoint"].to(self.device) if "extra_style_keypoint" in batch else None
        style_keypoint = torch.cat([keypoint, extra_style_keypoint], dim=2) if extra_style_keypoint is not None and keypoint is not None else keypoint

        keypoint_style, mu_logvar = None, None
        if self.use_keypoint_style_encoder:
            keypoint_style, mu_logvar = self.keypoint_style_encoder(style_keypoint)

        keypoint = self._apply_cfg_dropout(keypoint)

        disentanglement_key = batch.get("disentanglement_key", None)
        audio, keypoint = self._shuffle_for_disentanglement(audio, keypoint, disentanglement_key)

        conditioning = {"audio": audio}
        if keypoint is not None:
            conditioning["keypoint"] = keypoint
        if keypoint_style is not None:
            conditioning["keypoint_style"] = keypoint_style  # Add style feature to conditioning

        x_0 = torch.randn_like(x_1).to(self.device)
        t = torch.rand(x_1.shape[0], device=self.device)
        x_t, dx_t = self.path.interpolate(t=t, x_0=x_0, x_1=x_1)

        pred, _ = self.model(x_t, t, conditioning)

        loss, loss_dict = self._compute_loss(
            batch,
            pred,
            dx_t,
            disentanglement_key=disentanglement_key,
            template=batch.get("template", None),
            mu_logvar=mu_logvar,
        )

        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item(), loss_dict
