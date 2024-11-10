import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from encoders import ContextCNN_v2, State_Encoder_v3

from typing import Dict
from gym import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
  
class StateExtractor_v3(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, device='cuda:0', pos_enc:bool=False):
        super().__init__(observation_space, features_dim)

        self.state_encoder = State_Encoder_v3(
            hidden_size=56,
            lstm_input_size=6, 
            lstm_num_layers=1, 
            device=device,
            pos_enc=pos_enc
        )

    def forward(self, observations) -> th.Tensor:
        sv_states = observations['sv_states']
        ev_states = observations['ev_states']

        obs, _ = self.state_encoder(sv_states, ev_states)

        return obs
       
class CombinedExtractor_v3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, device='cuda:0', pos_enc:bool=False):
        super().__init__(observation_space, features_dim)
        
        self.state_encoder = State_Encoder_v3(
            hidden_size=56,
            lstm_input_size=6, 
            lstm_num_layers=1, 
            device=device,
            pos_enc=pos_enc
        )
        self.context_encoder = ContextCNN_v2(observation_space.spaces['context'], device=device)

    def forward(self, observations) -> th.Tensor:

        sv_states = observations['sv_states']
        ev_states = observations['ev_states']
        context = observations['context']   #1x8x256x256
        state_encoding, _ = self.state_encoder(sv_states, ev_states)      #1x128
        context_encoding = self.context_encoder(context)    #1x128
        final = th.cat((state_encoding, context_encoding), dim=-1)      #1x256

        return final

class CombinedExtractor_PPO_v2(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = ContextCNN_v2(observation_space.spaces['context'], features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)