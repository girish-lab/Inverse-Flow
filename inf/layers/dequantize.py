import torch

from .flowlayer import PreprocessingFlowLayer


class Dequantization(PreprocessingFlowLayer):
    def __init__(self, deq_distribution):
        super(Dequantization, self).__init__()
        # deq_distribution should be a distribution with support on [0, 1]^d
        self.distribution = deq_distribution

    def forward(self, input, context=None):
        # Note, input is the context for distribution model.
        noise, log_qnoise = self.distribution.sample(input.size(0), input.float())
        # print("Dequantization noise: ", noise.shape)
        # print("Dequantization log_qnoise: ", log_qnoise.shape)
        # print("Dequantization input: ", input.shape)
        # if torch.cuda.device_count() > 1:
        #     device = input.device
        #     noise = noise.to(device)
        #     log_qnoise = log_qnoise.to(device)
        # Convert log_qnoise to a tensor (if it's a float) and move it to the correct device
        device = input.device

        # If log_qnoise is a float, convert it into a tensor and move it to the device
        if isinstance(log_qnoise, float):
            log_qnoise = torch.tensor(log_qnoise, device=device)
        # If log_qnoise is a tensor, change it to float and move it to the device
        log_qnoise = log_qnoise.item() # if isinstance(log_qnoise, torch.Tensor) else log_qnoise
        noise = noise.to(device)
        # log_qnoise = log_qnoise.to(device)
        return input + noise, -log_qnoise

    def reverse(self, input, context=None):
        return input.floor()

    def logdet(self, input, context=None):
        raise NotImplementedError
