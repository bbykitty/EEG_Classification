from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

import torch
import abc



class AbstractDualInput(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(AbstractDualInput, self).__init__()
        
    @abc.abstractmethod
    def forward(self, x):
        pass

class Model_DICE_replica(AbstractDualInput):
    def __init__(self):
        super(Model_DICE_replica, self).__init__()
        #CNN
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19) 
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)

        #Positional Encoding
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))

        #CLS TOKEN
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 26, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 26, 1))

        #Transformer Enconder
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        #Feed Forward NN
        self.n_hidden=24
        self.layernorm = torch.nn.LayerNorm(normalized_shape=52)
        self.dropout1 = torch.nn.Dropout(.2)
        self.output = torch.nn.Linear(in_features=52, out_features=self.n_hidden)
        self.batchnorm1=torch.nn.BatchNorm1d(self.n_hidden)
        self.dropout2=torch.nn.Dropout(.2)
        self.output2=torch.nn.Linear(in_features=self.n_hidden,out_features=1)
        self.output3=torch.nn.Sigmoid()



    def forward(self, input1, input2):
        #Input Layer
        input1 = input1.permute(0,3,1,2)
        input2 = input2.permute(0,3,1,2)
        
        #CNN and Gelu
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze(dim=3)
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze(dim=3)
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)
        depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        #Positional Encoding Layer
        positional_enc1 = self.positional_encoding1(depthwise_conv_output1)
        positional_enc2 = self.positional_encoding1(depthwise_conv_output2)
        
        #Add CLS Token
        transformer_output1 =torch.cat([self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1],dim=2)
        transformer_output2=torch.cat([self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2],dim=2)

        #Transformer Encoder Layer
        transformer_output1 = self.transformer_encoder1(transformer_output1)
        transformer_output1 = transformer_output1[:,:,0]
        transformer_output2 = self.transformer_encoder2(transformer_output2)
        transformer_output2 = transformer_output2[:,:,0] 
        
        #Concat CLS Tokens
        concat_1_2 = torch.cat((transformer_output1, transformer_output2), dim=1)
        
        #Feed Forward
        #Layer normalization
        layer_norm_output = self.layernorm(concat_1_2)
        #Dropout
        layer_norm_output=self.dropout1(layer_norm_output)
        #Linear
        output = self.output(layer_norm_output)
        #Batch normalization
        output=self.batchnorm1(output)
        #Relu
        output=torch.nn.functional.relu(output)
        #Dropout
        output=self.dropout2(output)
        #Linear
        output2=self.output2(output)
        return output2

class Model_DICE_No_CNN(AbstractDualInput):
    def __init__(self):
        super(Model_DICE_No_CNN, self).__init__()
        #Positional Encoding
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))

        #CLS TOKEN
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 150, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 150, 1))

        #Transformer Enconder
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)

        #Feed Forward NN
        self.n_hidden=24
        self.layernorm = torch.nn.LayerNorm(normalized_shape=300)
        self.dropout1 = torch.nn.Dropout(.2)
        self.output = torch.nn.Linear(in_features=300, out_features=self.n_hidden)
        self.batchnorm1=torch.nn.BatchNorm1d(self.n_hidden)
        self.dropout2=torch.nn.Dropout(.2)
        self.output2=torch.nn.Linear(in_features=self.n_hidden,out_features=1)
        self.output3=torch.nn.Sigmoid()



    def forward(self, input1, input2):
        #Input layer
        input1 = input1.permute(0,3,1,2)
        input2 = input2.permute(0,3,1,2)
        input1 = torch.reshape(input1, (input1.shape[0], -1, 19))
        input2 = torch.reshape(input2, (input2.shape[0], -1, 19))

        #Positional Encoding Layer
        positional_enc1 = self.positional_encoding1(input1)
        positional_enc2 = self.positional_encoding1(input2)
        
        #Add CLS Token
        transformer_output1 =torch.cat([self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1],dim=2)
        transformer_output2=torch.cat([self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2],dim=2)

        #Transformer Encoder Layer
        transformer_output1 = self.transformer_encoder1(transformer_output1)
        transformer_output1 = transformer_output1[:,:,0]
        transformer_output2 = self.transformer_encoder2(transformer_output2)
        transformer_output2 = transformer_output2[:,:,0] 
        
        #Concat CLS Tokens
        concat_1_2 = torch.cat((transformer_output1, transformer_output2), dim=1)
        
        #Feed Forward
        #Layer normalization
        layer_norm_output = self.layernorm(concat_1_2)
        #Dropout
        layer_norm_output=self.dropout1(layer_norm_output)
        #Linear
        output = self.output(layer_norm_output)
        #Batch normalization
        output=self.batchnorm1(output)
        #Relu
        output=torch.nn.functional.relu(output)
        #Dropout
        output=self.dropout2(output)
        #Linear
        output2=self.output2(output)
        return output2

class Model_cls_late_concat(AbstractDualInput):
    def __init__(self):
        super(Model_cls_late_concat, self).__init__()
        ################# Depthwise convolution
        self.depth_conv1 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        self.depth_conv2 = torch.nn.Conv2d(in_channels=19, out_channels=19, kernel_size=(5,5), stride=(1,1), groups=19)
        ################# Positional Encoding (Sinsoidal)
        self.positional_encoding1 = Summer(PositionalEncoding1D(19))
        self.positional_encoding2 = Summer(PositionalEncoding1D(19))
        ######################CLS TOKEN NEW
        self.class_token1 = torch.nn.Parameter(torch.randn(1, 26, 1))
        self.class_token2 = torch.nn.Parameter(torch.randn(1, 26, 1))
        ################# Transformer Enconder Layer
        encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder1 = torch.nn.TransformerEncoder(encoder_layer1, num_layers=1)
        
        encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=20, nhead=2)
        self.transformer_encoder2 = torch.nn.TransformerEncoder(encoder_layer2, num_layers=1)
        ################# Output Layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=52)
        self.dropout1 = torch.nn.Dropout(0.20)
        self.output = torch.nn.Linear(in_features=52, out_features=24)
        self.batchnorm1=torch.nn.BatchNorm1d(24)
        #self.output2=torch.nn.Linear(in_features=16,out_features=8)
        self.dropout2=torch.nn.Dropout(0.20)
        #self.batchnorm2=torch.nn.BatchNorm1d(8)
        self.output3=torch.nn.Linear(in_features=24,out_features=1)
        self.dropout3 = torch.nn.Dropout(0.20)


    def forward(self, input1, input2):
        #fdsafgs
        input1 = input1.permute(0,3,1,2)                                                 # proper format (dimensions)
        input2 = input2.permute(0,3,1,2)
        # print("shapes")
        # print(input1.shape)
        ############################################################################
        depthwise_conv_output1 = self.depth_conv1(input1).squeeze(dim=3)                      # conv1
        depthwise_conv_output1 = torch.nn.functional.gelu(depthwise_conv_output1)
        # print("shapes")
        # print(depthwise_conv_output1.shape)
        depthwise_conv_output2 = self.depth_conv2(input2).squeeze(dim=3)                      # conv2
        depthwise_conv_output2 = torch.nn.functional.gelu(depthwise_conv_output2)
        
        ###permute conv1 and conv2
        depthwise_conv_output1=depthwise_conv_output1.permute(0,2,1)
        depthwise_conv_output2=depthwise_conv_output2.permute(0,2,1)
        
        positional_enc1=self.positional_encoding1(depthwise_conv_output1)
        positional_enc2=self.positional_encoding2(depthwise_conv_output2)
        transformer_output_all1=torch.cat((self.class_token1.expand(input1.shape[0],-1,-1),positional_enc1),dim=2)
        transformer_output_all1 = self.transformer_encoder1(transformer_output_all1)
        transformer_output_1 = transformer_output_all1[:,:,0]
        transformer_output_all2=torch.cat((self.class_token2.expand(input2.shape[0],-1,-1),positional_enc2),dim=2)
        transformer_output_all2 = self.transformer_encoder2(transformer_output_all2)
        transformer_output_2 = transformer_output_all2[:,:,0] 
        concat_1_2 = torch.cat((transformer_output_1, transformer_output_2), dim=1)
        ############################################################################
        layer_norm_output = self.layernorm(concat_1_2)
        ############################################################################
        layer_norm_output=self.dropout1(layer_norm_output)
        output = self.output(layer_norm_output)
        output=self.batchnorm1(output)
        output=torch.nn.functional.relu(output)
        output=self.dropout2(output)
        output3=self.output3(output)                                       # linear output
        return output3
        ############################################################################