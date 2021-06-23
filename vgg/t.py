# PyTorch中的torchvision里有很多常用的模型，可以直接调用：
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")
 
resnet101 = models.vgg16(pretrained=True)

