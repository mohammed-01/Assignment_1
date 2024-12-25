# Convulution weight = Cout x (K x K x Cin + 1)

import numpy as ng

class Conv:
    def __init__(self, kernel, bias=0, padding=0):
        self.kernel = ng.array(kernel)
        self.bias = bias
        self.padding = padding

    def operation(self, image):
        image = ng.array(image)
        if self.padding > 0:
            # Pad each dimension separately
            image = ng.pad(
                image,
                pad_width=((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode='constant',
                constant_values=0
            )
        kernel_height, kernel_width, input_channels, output_channels = self.kernel.shape
        image_height, image_width, _ = image.shape
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        convolved_image = ng.zeros((output_height, output_width, output_channels))

        for k in range(output_channels):
            for i in range(output_height):
                for j in range(output_width):
                    region = image[i:i + kernel_height, j:j + kernel_width, :]
                    convolved_image[i, j, k] = ng.sum(region * self.kernel[:, :, :, k]) + self.bias
        return convolved_image



class MaxPooling:
    def __init__(self, Pool_Size=(2, 2), stride=None):
        self.Pool_Size = Pool_Size
        self.stride = stride if stride is not None else Pool_Size  # default stride size is same as pool size

    def operation(self, image):
        image = ng.array(image)
        Pool_Height, Pool_Width = self.Pool_Size
        stride_y, stride_x = self.stride
        image_height, image_width, channels = image.shape
        Output_Height = (image_height - Pool_Height) // stride_y + 1
        output_width = (image_width - Pool_Width) // stride_x + 1
        pooled_image = ng.zeros((Output_Height, output_width, channels))
        # Performing Max Pooling
        for c in range(channels):
            for u in range(Output_Height):
                for i in range(output_width):
                    region = image[
                        u * stride_y : u * stride_y + Pool_Height,
                        i * stride_x : i * stride_x + Pool_Width,
                        c
                    ]
                    pooled_image[u, i, c] = ng.max(region)
        return pooled_image
