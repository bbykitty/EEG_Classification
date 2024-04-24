# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:56:15 2023

@author: AndreasMiltiadous
"""


from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

import torch
import abc



class AbstractDualInput(torch.nn.Module, abc.ABC):
    """
    Abstract class for all DICE models to inherit,
    so as to be in the same hierarchy
    """
    def __init__(self):
        super(AbstractDualInput, self).__init__()
        
    @abc.abstractmethod
    def forward(self, x):
        pass

class Model_DICE_replica(AbstractDualInput):
    def __init__(self):
        super(Model_DICE_replica, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19) 
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)

        ################# Positional Encoding
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))

        ######################CLS TOKEN NEW
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 26, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 26, 1))

        ################# Transformer Enconder Layer
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        ################# Output Layer
        self.n_hidden=24
        self.layernorm = torch.nn.LayerNorm(normalized_shape=52)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=52, out_features=self.n_hidden)
        self.batchnorm1=torch.nn.BatchNorm1d(self.n_hidden)
        self.dropout2=torch.nn.Dropout(0.20)
        self.output2=torch.nn.Linear(in_features=self.n_hidden,out_features=1)
        self.output3=torch.nn.Sigmoid()



    def forward(self, input1, input2):
        # print('input1', input1.shape)
        # print('input2', input2.shape)
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze(dim=3)                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze(dim=3)                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)               
        ###permute conv1 and conv2
        depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)                          # proper format (dimensions)
        depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        ############################################################################
        positional_enc1 = self.positional_encoding1(depthwise_conv_output1)  # positional encoding
        positional_enc2 = self.positional_encoding2(depthwise_conv_output2)
        ############################################################################
        transformer_output1 =torch.cat([self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1],dim=2)
        # print("cat shape: ", transformer_output1.shape)
        transformer_output1 = self.transformer_encoder1(transformer_output1)                # Transformer
        transformer_output1 = transformer_output1[:,:,0]
        # print("transformer shape: ", transformer_output1.shape)
        transformer_output2=torch.cat([self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2],dim=2)
        transformer_output2 = self.transformer_encoder2(transformer_output2)                # Transformer
        transformer_output2 = transformer_output2[:,:,0] 
        
        concat_1_2 = torch.cat((transformer_output1, transformer_output2), dim=1)
        ############################################################################
        # x=transformer_output_all.reshape(-1,26*39)
        layer_norm_output = self.layernorm(concat_1_2)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)

        output=torch.nn.functional.relu(output)

        output=self.dropout2(output)
        output2=self.output2(output)                                          # linear output
        return output2
        ############################################################################

class Model_DICE_No_CNN(AbstractDualInput):
    def __init__(self):
        super(Model_DICE_No_CNN, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19) 
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)

        ################# Positional Encoding
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))

        ######################CLS TOKEN NEW
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 150, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 150, 1))

        ################# Transformer Enconder Layer
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        ################# Output Layer
        self.n_hidden=24
        self.layernorm = torch.nn.LayerNorm(normalized_shape=300)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=300, out_features=self.n_hidden)
        self.batchnorm1=torch.nn.BatchNorm1d(self.n_hidden)
        self.dropout2=torch.nn.Dropout(0.20)
        self.output2=torch.nn.Linear(in_features=self.n_hidden,out_features=1)
        self.output3=torch.nn.Sigmoid()



    def forward(self, input1, input2):
        # print('input1', input1.shape)
        # print('input2', input2.shape)
        input1 = input1.permute(0,3,1,2)                                                # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        input1 = torch.reshape(input1, (input1.shape[0], -1, 19))
        input2 = torch.reshape(input2, (input2.shape[0], -1, 19))
        ############################################################################
        # depthwise_conv_output1 = self.depth_conv1(input1).squeeze(dim=3)                      # conv1
        # depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        # depthwise_conv_output2 = self.depth_conv2(input2).squeeze(dim=3)                      # conv2
        # depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)               
        ###permute conv1 and conv2
        # depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)                          # proper format (dimensions)
        # depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        ############################################################################
        positional_enc1 = self.positional_encoding1(input1)  # positional encoding
        positional_enc2 = self.positional_encoding2(input2)
        ############################################################################
        transformer_output1 =torch.cat([self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1],dim=2)
        # print("cat shape: ", transformer_output1.shape)
        transformer_output1 = self.transformer_encoder1(transformer_output1)                # Transformer
        transformer_output1 = transformer_output1[:,:,0]
        # print("transformer shape: ", transformer_output1.shape)
        transformer_output2=torch.cat([self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2],dim=2)
        transformer_output2 = self.transformer_encoder2(transformer_output2)                # Transformer
        transformer_output2 = transformer_output2[:,:,0] 
        
        concat_1_2 = torch.cat((transformer_output1, transformer_output2), dim=1)
        ############################################################################
        # x=transformer_output_all.reshape(-1,26*39)
        layer_norm_output = self.layernorm(concat_1_2)                         # layer normalization
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)

        output=torch.nn.functional.relu(output)

        output=self.dropout2(output)
        output2=self.output2(output)                                          # linear output
        return output2
        ############################################################################