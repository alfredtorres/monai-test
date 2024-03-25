# model = UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=2,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm=Norm.BATCH,
#     )
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection


class UNet5(nn.Module):
    def __init__(
            self,
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=2,
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act=Act.PRELU,
            norm=Norm.BATCH,
            dropout=0.0,
            bias=True,
            adn_ordering="NDA",
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        ## encoder
        self.encoder_layer0 = ResidualUnit(self.dimensions, in_channels, channels[0], strides=strides,
                                           kernel_size=self.kernel_size, subunits=self.num_res_units,
                                           act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                           adn_ordering=self.adn_ordering)
        self.encoder_layer1 = ResidualUnit(self.dimensions, channels[0], channels[1], strides=strides,
                                           kernel_size=self.kernel_size, subunits=self.num_res_units,
                                           act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                           adn_ordering=self.adn_ordering)
        self.encoder_layer2 = ResidualUnit(self.dimensions, channels[1], channels[2], strides=strides,
                                           kernel_size=self.kernel_size, subunits=self.num_res_units,
                                           act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                           adn_ordering=self.adn_ordering)
        self.encoder_layer3 = ResidualUnit(self.dimensions, channels[2], channels[3], strides=strides,
                                           kernel_size=self.kernel_size, subunits=self.num_res_units,
                                           act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                           adn_ordering=self.adn_ordering)
        self.encoder_layer4 = ResidualUnit(self.dimensions, channels[3], channels[4], strides=1,
                                           kernel_size=self.kernel_size, subunits=self.num_res_units,
                                           act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                           adn_ordering=self.adn_ordering)
        ## decoder
        self.decoder_layer4_conv1 = Convolution(self.dimensions, channels[4] + channels[3], channels[2],
                                                strides=strides,
                                                kernel_size=self.up_kernel_size, act=self.act,
                                                norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                conv_only=False and self.num_res_units == 0, is_transposed=True,
                                                adn_ordering=self.adn_ordering)
        self.decoder_layer4_conv2 = ResidualUnit(self.dimensions, channels[2], channels[2], strides=1,
                                                 kernel_size=self.kernel_size, subunits=1,
                                                 act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                 last_conv_only=False, adn_ordering=self.adn_ordering)

        self.decoder_layer3_conv1 = Convolution(self.dimensions, channels[2] * 2, channels[1], strides=strides,
                                                kernel_size=self.up_kernel_size, act=self.act,
                                                norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                conv_only=False and self.num_res_units == 0, is_transposed=True,
                                                adn_ordering=self.adn_ordering)
        self.decoder_layer3_conv2 = ResidualUnit(self.dimensions, channels[1], channels[1], strides=1,
                                                 kernel_size=self.kernel_size, subunits=1,
                                                 act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                 last_conv_only=False, adn_ordering=self.adn_ordering)

        self.decoder_layer2_conv1 = Convolution(self.dimensions, channels[1] * 2, channels[0], strides=strides,
                                                kernel_size=self.up_kernel_size, act=self.act,
                                                norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                conv_only=False and self.num_res_units == 0, is_transposed=True,
                                                adn_ordering=self.adn_ordering)
        self.decoder_layer2_conv2 = ResidualUnit(self.dimensions, channels[0], channels[0], strides=1,
                                                 kernel_size=self.kernel_size, subunits=1,
                                                 act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                 last_conv_only=False, adn_ordering=self.adn_ordering)

        self.decoder_layer1_conv1 = Convolution(self.dimensions, channels[0] * 2, self.out_channels, strides=strides,
                                                kernel_size=self.up_kernel_size, act=self.act,
                                                norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                conv_only=True and self.num_res_units == 0, is_transposed=True,
                                                adn_ordering=self.adn_ordering)
        self.decoder_layer1_conv2 = ResidualUnit(self.dimensions, self.out_channels, self.out_channels, strides=1,
                                                 kernel_size=self.kernel_size, subunits=1,
                                                 act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                                 last_conv_only=True, adn_ordering=self.adn_ordering)

        # self.decoder_layer0_conv1 = Convolution(self.dimensions, channels[1], channels[0], strides=strides,
        #                                         kernel_size=self.up_kernel_size, act=self.act,
        #                                         norm=self.norm, dropout=self.dropout, bias=self.bias,
        #                                         conv_only=True and self.num_res_units == 0, is_transposed=True,
        #                                         adn_ordering=self.adn_ordering)
        # self.decoder_layer0_conv2 = ResidualUnit(self.dimensions, channels[0], channels[0], strides=1,
        #                                          kernel_size=self.kernel_size, subunits=1,
        #                                          act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
        #                                          last_conv_only=True, adn_ordering=self.adn_ordering)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N,N,N,1)
        x0 = self.encoder_layer0(x)  # (N/2,N/2,N/2,16)
        x1 = self.encoder_layer1(x0)  # (N/4,N/4,N/4,32)
        x2 = self.encoder_layer2(x1)  # (N/8,N/8,N/8,64)
        x3 = self.encoder_layer3(x2)  # (N/16,N/16,N/16,128)
        x4 = self.encoder_layer4(x3)  # (N/16,N/16,N/16,256)

        y3 = torch.cat((x4, x3), dim=1)
        y3 = self.decoder_layer4_conv1(y3)  # (N/8,N/8,N/8,64)
        y3 = self.decoder_layer4_conv2(y3)

        y2 = torch.cat((y3, x2), dim=1)
        y2 = self.decoder_layer3_conv1(y2)  # (N/4,N/4,N/4,32)
        y2 = self.decoder_layer3_conv2(y2)

        y1 = torch.cat((y2, x1), dim=1)
        y1 = self.decoder_layer2_conv1(y1)  # (N/2,N/2,N/2,16)
        y1 = self.decoder_layer2_conv2(y1)

        y0 = torch.cat((y1, x0), dim=1)
        y0 = self.decoder_layer1_conv1(y0)  # (N,N,N,2)
        y0 = self.decoder_layer1_conv2(y0)
        return y0

if __name__ == '__main__':
    model = UNet5()
    inputs = torch.randn(1, 1, 96, 96, 96)
    outputs = model(inputs)
    print(model)
    # torch.save(model, "./UNet5_viz.pt")
