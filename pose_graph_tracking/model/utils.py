import torch

from torch.nn import Module, Linear as Lin, LayerNorm, Conv2d, ReLU, LeakyReLU, Sigmoid, Tanh


class Flatten(Module):
    """
    Copied from pytorch 1.2.0
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(self.start_dim, self.end_dim)


def get_activation_function_from_type(activation_type: str):
    available_activation_types = ["relu", "leaky_relu", "sigmoid", "tanh"]
    assert activation_type in available_activation_types, "Requested activation type %r is not an available " \
                                                          "activation type: %r" % (activation_type,
                                                                                   available_activation_types)
    if activation_type == "relu":
        return ReLU
    elif activation_type == "leaky_relu":
        return LeakyReLU
    elif activation_type == "sigmoid":
        return Sigmoid
    elif activation_type == "tanh":
        return Tanh


def init_weights(module: Module,
                 activation_type: str = "relu"):
    negative_slope = 0 if activation_type == "relu" else 1e-2
    if type(module) == Lin or type(module) == Conv2d:
        torch.nn.init.kaiming_uniform_(module.weight, a=negative_slope, nonlinearity=activation_type)


def generate_encoder(number_of_input_channels: int,
                     number_of_hidden_channels: int,
                     number_of_output_channels: int,
                     activation_function: Module,
                     number_of_hidden_layers: int = 1) -> torch.nn.Sequential:
    """
    Generates an encoder network wrt. input parameters.

    Exemplary encoder with 1 hidden layer and ReLu as activation function:

    Input layer:
    Lin(number_of_input_channels, number_of_hidden_channels)

    Hidden layer:
    ReLu()
    Lin(number_of_hidden_channels, number_of_hidden_channels)

    Output layer:
    ReLu()
    Lin(number_of_hidden_channels, number_of_output_channels)
    LayerNorm(number_of_outputs)

    :param number_of_input_channels: Number of input features/channels.
    :param number_of_hidden_channels: Number of hidden features/channels.
    :param number_of_output_channels: Number of output features/channels.
    :param activation_function: Activation function.
    :param number_of_hidden_layers: Number of hidden layers used for the encoder.
    :return: Returns the generated encoder.
    """
    assert number_of_input_channels > 0, "Number of input channels for encoder must be a positive int."
    assert number_of_hidden_channels > 0, "Number of hidden channels for encoder must be a positive int."
    assert number_of_output_channels > 0, "Number of output channels for encoder must be a positive int."
    assert number_of_hidden_layers >= 0, "Number of hidden layers for encoder must be a positive int or zero."

    sequential = torch.nn.Sequential()
    sequential.add_module("encoder_input_layer", Lin(number_of_input_channels, number_of_output_channels))

    for layer_id in range(number_of_hidden_layers):
        sequential.add_module("encoder_activation_function_" + str(layer_id), activation_function())
        sequential.add_module("encoder_hidden_layer_" + str(layer_id), Lin(number_of_hidden_channels,
                                                                           number_of_hidden_channels))

    sequential.add_module("encoder_activation_function_" + str(number_of_hidden_layers), activation_function())
    sequential.add_module("encoder_output_layer", Lin(number_of_hidden_channels, number_of_output_channels))
    sequential.add_module("encoder_layer_norm", LayerNorm(number_of_output_channels))
    sequential.apply(init_weights)
    return sequential


def generate_decoder(number_of_input_channels: int,
                     number_of_hidden_channels: int,
                     number_of_output_channels: int,
                     activation_function: Module,
                     number_of_hidden_layers: int = 1) -> torch.nn.Sequential:
    """
    Generates an decoder network wrt. input parameters.

    Exemplary decoder with 1 hidden layer and ReLu as activation function:

    Input layer:
    Lin(number_of_input_channels, number_of_hidden_channels)
    ReLu()

    Hidden layer:
    Lin(number_of_hidden_channels, number_of_hidden_channels)
    ReLu()

    Output layer:
    Lin(number_of_hidden_channels, number_of_output_channels)

    :param number_of_input_channels: Number of input features/channels.
    :param number_of_hidden_channels: Number of hidden features/channels.
    :param number_of_output_channels: Number of output features/channels.
    :param activation_function: Activation function.
    :param number_of_hidden_layers: Number of hidden layers used for the decoder.
    :return: Returns the generated decoder.
    """
    assert number_of_input_channels > 0, "Number of input channels for decoder must be a positive int."
    assert number_of_hidden_channels > 0, "Number of hidden channels for decoder must be a positive int."
    assert number_of_output_channels > 0, "Number of output channels for decoder must be a positive int."
    assert number_of_hidden_layers >= 0, "Number of hidden layers for decoder must be a positive int or zero."

    sequential = torch.nn.Sequential()
    sequential.add_module("decoder_input", Lin(number_of_input_channels, number_of_hidden_channels))
    sequential.add_module("decoder_input_activation_function", activation_function())

    for layer_id in range(number_of_hidden_layers):
        sequential.add_module("decoder_hidden_" + str(layer_id), Lin(number_of_hidden_channels,
                                                                     number_of_hidden_channels))
        sequential.add_module("decoder_activation_function_" + str(layer_id), activation_function())

    sequential.add_module("decoder_output", Lin(number_of_hidden_channels, number_of_output_channels))
    sequential.apply(init_weights)
    return sequential
