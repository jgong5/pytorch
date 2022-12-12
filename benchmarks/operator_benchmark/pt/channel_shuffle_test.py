import operator_benchmark as op_bench
import torch


"""Microbenchmarks for channel_shuffle operator."""


# Configs for PT channel_shuffle operator
channel_shuffle_long_configs = op_bench.cross_product_configs(
    batch_size=[4, 8],
    channels_per_group=[32, 64],
    height=[32, 64],
    width=[32, 64],
    groups=[4, 8],
    channel_last=[True, False],
    tags=["long"]
)


channel_shuffle_short_configs = op_bench.config_list(
    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],
    attrs=[
        [64, 58, 28, 28, 2],
    ],
    cross_product_configs={
        "channel_last": [True],
    },
    tags=["short"]
)

from torch import Tensor
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        channels = channels_per_group * groups
        data_shape = (batch_size, channels, height, width)
        input_data = torch.rand(data_shape)
        if channel_last:
            input_data = input_data.contiguous(memory_format=torch.channels_last)
        self.inputs = {
            "input_data": input_data,
            "groups": groups
        }
        self.set_module_name('channel_shuffle')

    def compute1(self, x1):
        x1 += 1
        return x1;

    def compute2(self, x2):
        x2 += 2
        return x2;

    def forward(self, input_data, groups: int):
        x1, x2 = input_data.chunk(2, dim=1)
        input_data = torch.cat([self.compute1(x1), self.compute2(x2)], dim=1)
        return channel_shuffle(input_data, groups)

class ChannelSHuffleBenchmarkTI(ChannelSHuffleBenchmark):
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        super().init(batch_size, channels_per_group, height, width, groups, channel_last)

    @torch.compile()
    def forward(self, input_data, groups: int):
        return super().forward(input_data, groups)

op_bench.generate_pt_test(channel_shuffle_short_configs + channel_shuffle_long_configs,
                          ChannelSHuffleBenchmark)

op_bench.generate_pt_test(channel_shuffle_short_configs + channel_shuffle_long_configs,
                          ChannelSHuffleBenchmarkTI)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
