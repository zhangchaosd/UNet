from torchvision import transforms

class ImageTransform(object):
    def __init__(self, input_size = 572, needCrop = False, cropSize = 388):
        assert isinstance(input_size, int)
        self.input_size = input_size
        self.needCrop = needCrop
        self.cropSize = cropSize

    def __call__(self, img):
        h, w = img.shape[1:] # 3, h, w
        paddinglr = 0
        paddingud = 0 
        if h < w:
            paddingud = (w - h) // 2
        elif w < h:
            paddinglr = (h - w) // 2
        img = transforms.functional.pad(img, padding = (paddinglr, paddingud), padding_mode = 'reflect')
        img = transforms.functional.resize(img, (self.input_size, self.input_size))
        if self.needCrop:
            img = transforms.functional.center_crop(img, self.cropSize)
        return img
