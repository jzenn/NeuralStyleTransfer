import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models

import copy

from utils import compute_i_th_moment
from utils import gram_matrix
from utils import linear_time_mmd

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization mean and std
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def get_vgg_model(configuration):
    """
    get the pre-trained VGG-19 model from the PyTorch framework
    :param configuration: the config file
    :return: the pre-trained VGG-19 model
    """
    vgg_pre_trained = models.vgg19()
    vgg_pre_trained_state_dict = torch.utils.model_zoo.load_url(configuration['model_url'],
                                                                model_dir=configuration['model_dir'])
    vgg_pre_trained.load_state_dict(vgg_pre_trained_state_dict)
    vgg_pre_trained = vgg_pre_trained.features.to(device).eval()
    return vgg_pre_trained


def get_style_loss_module(i):
    """
    returns the style loss module corresponding to i
    :param i:
    :return:
    """
    if i == 1:
        return StyleLossGramMatrix
    # MMD loss does not produce the expected results
    # (within two corresponding feature maps and not between all feature maps, entry-wise)
    # elif i == 2:
    #    return StyleLossMMD
    elif i == 2:
        return StyleLossMean
    elif i == 3:
        return StyleLossMeanStd
    elif i == 4:
        return StyleLossMeanStdSkew
    elif i == 5:
        return StyleLossMeanStdSkewKurtosis
    else:
        raise RuntimeError('could not recognize i = {}'.format(i))


def get_content_loss_module():
    return ContentLoss


def get_full_style_model(configuration, vgg_model, style_image, content_image, style_loss_module, content_loss_module):
    """
    produce the full model from the pre-trained VGG-19 model
    :param configuration:
    :param vgg_model:
    :param style_image:
    :param content_image:
    :param style_loss_module:
    :param content_loss_module:
    :return:
    """
    vgg_model = copy.deepcopy(vgg_model)

    # style layers
    style_layers = configuration['style_layers']

    # content layers
    content_layers = configuration['content_layers']

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # the style losses
    style_losses = []

    # the content losses
    content_losses = []

    model = nn.Sequential(normalization)

    i = 1
    j = 1
    for layer in vgg_model.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}_{}'.format(i, j)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            j = 1
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}_{}'.format(i, j)
            # this inplace=False is very important, otherwise PyTorch throws an exception
            # ('one of the variables needed for gradient computation has been modified by an inplace operation')
            layer = nn.ReLU(inplace=False)
            j += 1
        else:
            raise RuntimeError('unrecognized layer')

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_image).detach()

            style_loss = style_loss_module(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in content_layers:
            # add style loss:
            target_feature = model(content_image).detach()

            content_loss = content_loss_module(target_feature)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], style_loss_module) or isinstance(model[i], content_loss_module):
            break

    # can totally trim of the model after the last style loss layer
    model = model[:(i + 1)]

    print(model)

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer


class Normalization(nn.Module):
    """
    normalization module to normalize the image data with mean and std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleLossMMD(nn.Module):
    """
    Style Loss MMD
    """
    def __init__(self, target_feature, alpha=1/10):
        super(StyleLossMMD, self).__init__()
        self.target = target_feature.detach()
        self.alpha = alpha
        self.loss = 0

    def forward(self, input):
        self.loss = linear_time_mmd(input, self.target, self.alpha)
        return input


class ContentLoss(nn.Module):
    """
    Content Loss
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.loss = 0
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLossGramMatrix(nn.Module):
    """
    Style Loss Gram Matrix
    """
    def __init__(self, target_feature):
        super(StyleLossGramMatrix, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input


class StyleLossMeanStdSkewKurtosis(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std, Skew, Kurt
    """
    def __init__(self, target):
        super(StyleLossMeanStdSkewKurtosis, self).__init__()
        self.target_mean = torch.abs(compute_i_th_moment(target, 1))
        self.target_std = torch.abs(compute_i_th_moment(target, 2))
        self.target_skewness = torch.abs(compute_i_th_moment(target, 3))
        self.target_kurtosis = torch.abs(compute_i_th_moment(target, 4))

        self.loss = 0

    def forward(self, input):
        input_mean = torch.abs(compute_i_th_moment(input, 1))
        input_std = torch.abs(compute_i_th_moment(input, 2))
        input_skewness = torch.abs(compute_i_th_moment(input, 3))
        input_kurtosis = torch.abs(compute_i_th_moment(input, 4))

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))
        skew_balancing_factor = torch.min(1 / torch.mean(self.target_skewness).to(device), torch.tensor(1.).to(device))
        kurtosis_balancing_factor = torch.min(1 / torch.mean(self.target_kurtosis).to(device), torch.tensor(1.).to(device))

        self.loss = torch.abs(mean_balancing_factor) * F.mse_loss(input_mean.to(device), self.target_mean.to(device))           \
                  + torch.abs(std_balancing_factor) * F.mse_loss(input_std.to(device), self.target_std.to(device))              \
                  + torch.abs(skew_balancing_factor) * F.mse_loss(input_skewness.to(device), self.target_skewness.to(device))   \
                  + torch.abs(kurtosis_balancing_factor) * F.mse_loss(input_kurtosis.to(device), self.target_kurtosis.to(device))

        return input


class StyleLossMeanStdSkew(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std, Skew
    """
    def __init__(self, target):
        super(StyleLossMeanStdSkew, self).__init__()
        self.target_mean = torch.abs(compute_i_th_moment(target, 1))
        self.target_std = torch.abs(compute_i_th_moment(target, 2))
        self.target_skewness = torch.abs(compute_i_th_moment(target, 3))

        self.loss = 0

    def forward(self, input):
        input_mean = torch.abs(compute_i_th_moment(input, 1))
        input_std = torch.abs(compute_i_th_moment(input, 2))
        input_skewness = torch.abs(compute_i_th_moment(input, 3))

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))
        skew_balancing_factor = torch.min(1 / torch.mean(self.target_skewness).to(device), torch.tensor(1.).to(device))

        self.loss = torch.abs(mean_balancing_factor) * F.mse_loss(input_mean.to(device), self.target_mean.to(device)) \
                  + torch.abs(std_balancing_factor) * F.mse_loss(input_std.to(device), self.target_std.to(device)) \
                  + torch.abs(skew_balancing_factor) * F.mse_loss(input_skewness.to(device), self.target_skewness.to(device))

        return input


class StyleLossMeanStd(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std
    """
    def __init__(self, target):
        super(StyleLossMeanStd, self).__init__()
        self.target_mean = torch.abs(compute_i_th_moment(target, 1))
        self.target_std = torch.abs(compute_i_th_moment(target, 2))

        self.loss = 0

    def forward(self, input):
        input_mean = torch.abs(compute_i_th_moment(input, 1))
        input_std = torch.abs(compute_i_th_moment(input, 2))

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))

        self.loss = torch.abs(mean_balancing_factor) * F.mse_loss(input_mean.to(device), self.target_mean.to(device)) \
                  + torch.abs(std_balancing_factor) * F.mse_loss(input_std.to(device), self.target_std.to(device))

        return input


class StyleLossMean(nn.Module):
    """
    Style Loss (Moment Loss) - Mean
    """
    def __init__(self, target):
        super(StyleLossMean, self).__init__()
        self.target_mean = torch.abs(compute_i_th_moment(target, 1))

        self.loss = 0

    def forward(self, input):
        input_mean = torch.abs(compute_i_th_moment(input, 1))

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))

        self.loss = torch.abs(mean_balancing_factor) * F.mse_loss(input_mean.to(device), self.target_mean.to(device))

        return input