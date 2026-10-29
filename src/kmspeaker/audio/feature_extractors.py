# External Modules
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor
from transformers import AutoModel


class AudioFeatureExtractor(nn.Module):
    """
    Loads a pre-trained Wav2Vec2BertModel and extracts its raw hidden states.
    """

    def __init__(self, model_name="facebook/w2v-bert-2.0", cache_dir=None):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

        self.num_hidden_layers = self.model.config.num_hidden_layers + 1
        self.n_hidden_dim = self.model.config.hidden_size

    def forward(self, raw_audio_batch, sampling_rate):
        """
        Extracts hidden states from a batch of raw audio.

        Args:
            raw_audio_batch (list of np.array or torch.Tensor): Batch of raw audio signals.
                                                                Each element is a 1D array/tensor.
            sampling_rate (int): The sampling rate of the audio.

        Returns:
            torch.Tensor: Stacked hidden states (N x T_audio_fps x n_hidden_layers x n_hidden_dim).
        """
        raw_audio_batch = [speech.cpu() if isinstance(speech, torch.Tensor) else speech for speech in raw_audio_batch]

        inputs = self.processor(raw_audio_batch, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # We stack them along a new dimension to get (batch_size, sequence_length, num_layers, hidden_size)
        stacked_hidden_states = torch.stack(outputs.hidden_states, dim=2)

        if stacked_hidden_states.shape[2] != self.num_hidden_layers:
            raise ValueError(f"Mismatch in expected number of hidden layers. Expected {self.num_hidden_layers}, got {stacked_hidden_states.shape[2]}")

        return stacked_hidden_states
