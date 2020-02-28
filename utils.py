import torch

import torchvision.utils as utils
import numpy.random as rand

import datetime
import pytz

# the date-time format
fmt = '%d_%m__%H_%M_%S'


def compute_i_th_moment(input, i):
    """
    computes the i-th moment of the input tensor channel-wise
    :param input: tensor of size (n, c, h, w), n=1
    :param i: the moment one wants to compute
    :return: tensor with the i-th moment of every channel
    """
    # get the input size
    input_size = input.size()

    # (n, c, h, w)
    n = input_size[0]
    c = input_size[1]

    mean = torch.mean(input.view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)

    eps = 1e-5
    var = torch.var(input.view(n, c, -1), dim=2, keepdim=True) + eps
    std = torch.sqrt(var).view(n, c, 1, 1)

    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        return torch.mean((((input - mean) / std).pow(i)).view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)


def gram_matrix(input):
    """
    gets the gram matrix of the input and normalizes it
    :param input: a feature tensor of size (n, c, h, w)
    :return the gram matrix
    """
    n, c, h, w = input.size()
    features = input.view(n * c, h * w)
    gram = torch.mm(features, features.t())  # compute the Gram product

    # normalize Gram matrix
    return gram.div(n * c * h * w)


def split_even_odd(x):
    """
    split a list into two different lists by the even and odd entries
    :param x: the list
    :return: two lists with even and odd entries of x respectively
    """
    n, c, h, w = x.size()
    x = x.view(n * c, -1)
    return [x[0:((n * c) - ((n * c) % 2)), :].t().view(h * w, -1, 2)[:, :, 1],
            x[0:((n * c) - ((n * c) % 2)), :].t().view(h * w, -1, 2)[:, :, 0]]


def gaussian_kernel(x, y, alpha):
    """
    compute a Gaussian kernel for vector x and y
    :param x: data list
    :param y: data list
    :param alpha: parameter for the Gaussian kernel
    :return: the Gaussian kernel
    """
    # e^(-a * |x-y|^2)
    return torch.exp(-alpha * torch.sum((x - y).pow(2), dim=1))


def mmd_polynomial_kernel(x, y, alpha=2):
    """
    compute a polynomial kernel for vector x and y
    :param x: data list
    :param y: data list
    :param alpha: parameter for the polynomial kernel
    :return: the polynomial kernel
    """
    n, c, h, w = x.size()
    x = x.view(n * c, h * w)
    y = y.view(n * c, h * w)
    return 1 / (n * c * h * w) * torch.sum(torch.sum(x * y, dim=0).pow(alpha))


def h(x_i, y_i, x_j, y_j, alpha):
    """
    helper function for the MMD O(n) computation
    :param x_i: odd entries of x
    :param y_i: odd entries of y
    :param x_j: even entries of x
    :param y_j: even entries of y
    :param alpha: the parameter for the Gaussian kernel
    :return: the value for the Gaussian kernel
    """
    # compute kernel values
    s1 = gaussian_kernel(x_i, x_j, alpha)
    s2 = gaussian_kernel(y_i, y_j, alpha)
    s3 = gaussian_kernel(x_i, y_j, alpha)
    s4 = gaussian_kernel(x_j, y_i, alpha)

    return torch.sum(s1) + torch.sum(s2) - torch.sum(s3) - torch.sum(s4)


def linear_time_mmd(x, y, alpha):
    """
    compute the linear time O(n) approximation of the MMD
    :param x:
    :param y:
    :param alpha:
    :return:
    """
    # split tensors x and y channel-wise based on its index
    x_even, x_odd = split_even_odd(x)
    y_even, y_odd = split_even_odd(y)

    # number of even/odd elements
    _, result_length = x_even.size()

    # return mmd approximation
    return torch.abs(1 / result_length * h(x_odd, y_odd, x_even, y_even, alpha))


def save_image(configuration, images, style_image_number, content_image_number):
    """
    save the images in several compositions
    :param configuration: the config file
    :param images: list of images
    :param style_image_number: number of the style image
    :param content_image_number: number of the content image
    :return:
    """
    image_saving_path = configuration['image_saving_path']
    style_image_folder = configuration['style_image_folder']
    content_image_folder = configuration['content_image_folder']

    print('print saving transfer image of style {} and content {}'.format(style_image_number, content_image_number))
    # | content | style   | white   | Gram     |
    # | moment1 | moment2 | moment3 | moment 4 |
    utils.save_image(images[0:2]
                     + [torch.ones(images[0].size()).cpu()]
                     + images[2:],
                     filename='{}/A_style_image_{}__content_image_{}__{}_{}__{}__{}.jpeg'.format(
                         image_saving_path,
                         style_image_number,
                         content_image_number,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt),
                         rand.randint(low=0, high=100),
                         style_image_folder,
                         content_image_folder), nrow=4, pad_value=1, normalize=False)

    # | Gram     | moment1 | moment2 | moment3 | moment 4 |
    utils.save_image(images[2:],
                     filename='{}/B_style_image_{}__content_image_{}__{}_{}__{}__{}.jpeg'.format(
                         image_saving_path,
                         style_image_number,
                         content_image_number,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt),
                         rand.randint(low=0, high=100),
                         style_image_folder,
                         content_image_folder), pad_value=1, normalize=False)

    # | style   | moment1 | moment2 | moment3 | moment 4 |
    utils.save_image([images[0]]
                     + images[3:],
                     filename='{}/C_style_image_{}__content_image_{}__{}_{}__{}__{}.jpeg'.format(
                         image_saving_path,
                         style_image_number,
                         content_image_number,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt),
                         rand.randint(low=0, high=100),
                         style_image_folder,
                         content_image_folder), pad_value=1, normalize=False)

    # | content | moment1 | moment2 | moment3 | moment 4 |
    utils.save_image([images[1]]
                     + images[3:],
                     filename='{}/D_style_image_{}__content_image_{}__{}_{}__{}__{}.jpeg'.format(
                         image_saving_path,
                         style_image_number,
                         content_image_number,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt),
                         rand.randint(low=0, high=100),
                         style_image_folder,
                         content_image_folder), pad_value=1, normalize=False)


def save_single_image(configuration, image, style_image_number, content_image_number):
    """
    save the single image to the saving path
    :param configuration: the config file
    :param image: the image to be saved
    :param style_image_number: the style image number
    :param content_image_number: the content image number
    :return: 
    """
    image_saving_path = configuration['image_saving_path']
    style_image_folder = configuration['style_image_folder']
    content_image_folder = configuration['content_image_folder']

    loss_names = {
        -1: 'Gram',
        -2: 'Mean',
        -3: 'Std',
        -4: 'Skew',
        -5: 'Kurt'
    }

    try:
        loss_name = loss_names[style_image_number]
    except:
        loss_name = 'err'

    print('print saving single image of {} and {}'.format(style_image_number, content_image_number))
    utils.save_image(image,
                     filename='{}/single_style_image_{}__content_image_{}__{}_{}_{}_{}__{}__{}.jpeg'.format(
                         image_saving_path,
                         style_image_number,
                         content_image_number,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt),
                         rand.randint(low=0, high=100),
                         rand.randint(low=0, high=100),
                         loss_name,
                         style_image_folder,
                         content_image_folder), pad_value=1, normalize=False)


