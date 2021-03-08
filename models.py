from torch import nn

class OneUpOneDownAutoencoder(nn.Module):
    
    def __init__(self, in_channels, 
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding=0):
        super(OneUpOneDownAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, 
                      out_channels = self.out_channels,
                      kernel_size = kernel_size, 
                      stride = stride, 
                      padding = padding), 
            nn.ReLU(True),
            # Maybe add pooling here
        ) # output shape = (N, out_channels, H*, W*)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = self.out_channels, 
                out_channels = self.in_channels,
                kernel_size = kernel_size, 
                stride=stride,
                padding = padding,
                output_padding=padding),
            nn.ReLU(True),
        )  # output shape = (N, in_channels, H, W)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class SimpleDecoderLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, 
                 kernel_size = 3,
                 stride=1,
                 padding=0):
        super(SimpleDecoderLayer, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, 
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                output_padding = padding),
            nn.ReLU(True),
        )  # output shape = (N, in_channels, H, W)
        
    def forward(self, x):
        return self.decoder(x)

class HierarchicalAutoencoder(nn.Module):
    
    class ModuleListForward(nn.ModuleList):
        """
        Module list with forward function
        """
        def forward(self, x):
            for module in self:
                x = module(x)
            return x
            
    
    def __init__(self, encoder_layer_func, 
                 decoder_layer_func,
                 num_layers,
                 input_size,
                 output_sizes = [],
                 stride = 1,
                 padding = 0):
        super(HierarchicalAutoencoder, self).__init__()
        
        if type(output_sizes) is int:
            output_sizes = [output_sizes]
            
        
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            if i >= len(output_sizes):
                output_size = input_size * 2
            else:
                output_size = output_sizes[i]
                
            instantiated_encoder = encoder_layer_func(
                in_channels = input_size,
                out_channels = output_size,
                stride = stride,
                padding = padding
            )
            self.encoder_layers.append(instantiated_encoder)
            
            input_size = output_size
    
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = self.encoder_layers[num_layers - i - 1].out_channels
            output_size = self.encoder_layers[num_layers - i - 1].in_channels
            instantiated_decoder = decoder_layer_func(
                in_channels = input_size,
                out_channels = output_size,
                stride = stride,
                padding =padding
            )
            self.decoder_layers.append(instantiated_decoder)
        
        #self.encoder = nn.Sequential(*self.encoder_layers)
        #self.decoder = nn.Sequential(*self.decoder_layers)
            
    def forward(self, x):
        for layer in self.encoder_layers:
            x, _ = layer(x)
            
        encoded = x
        for layer in self.decoder_layers:
            x = layer(x)
            
        decoded = x
        return encoded, decoded
    
    def get_extractor(self, layer_num=None):
        """
        Return the feature extractor at the "layer_num" layer
        """
        if layer_num is None:
            layer_num = len(self.encoder_layers)
        
        if layer_num > len(self.encoder_layers):
            raise ValueError("Asked for too many layers for extractor")
            
        extractor = nn.ModuleList()
        for i in range(layer_num):
            extractor.append(self.encoder_layers[i].encoder)
        
        return self.ModuleListForward(extractor)
