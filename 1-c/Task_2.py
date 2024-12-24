# Convulution weight = Cout x (K x K x Cin + 1)

import numpy as ng

class Conv:
    def __init__(self,kernel,bias = 0,padding = 0):

        self.kernel = ng.array(kernel)
        self.bias = bias
        self.padding = padding

    def operation(self,image):

        image = ng.array(image)
        if self.padding > 0:
            image = ng.pad(
                image,
                pad_width=((self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0)
        kernel_height, kernel_width = self.kernel.shape
        image_height, image_width = image.shape
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        convolved_image = ng.zeros((output_height, output_width))
        # Perform convolution
        for i in range(output_height):
            for j in range(output_width):
                region = image[i:i + kernel_height, j:j + kernel_width]
                convolved_value = ng.sum(region * self.kernel)
                convolved_image[i, j] = convolved_value + self.bias
        return convolved_image


class MaxPooling:
    def __init__(self, Pool_Size=(2, 2), Stride= None):

        self.Pool_Size = Pool_Size
        self.stride = Stride if Stride is not None else Pool_Size  #default stride size is same as pool size

    def operation(self, image):

        image = ng.array(image)
        Pool_Hieght, Pool_Width = self.Pool_Size #Hieght?!
        Stride_y, Stride_x = self.Stride
        image_height, image_width = image.shape

        Output_Height = (image_height - Pool_Hieght) // Stride_y + 1
        output_width = (image_width - Pool_Width) // Stride_x + 1

        pooled_image = ng.zeros((Output_Height, output_width))

        #Performing Max Pooling
        for u in range(Output_Height):
            for i in range(output_width):
                region = image[
                    u * Stride_y : u * Stride_y + Pool_Hieght,
                    i * Stride_x : i * Stride_x + Pool_Width
                ]
                pooled_image[u, i] = ng.max(region)

        return pooled_image