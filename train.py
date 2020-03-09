import os
import sys
import torch

from net import get_vgg_model
from net import get_style_loss_module
from net import get_content_loss_module
from net import get_full_style_model
from net import get_input_optimizer

from data_loader import get_images
from data_loader import load_image

from utils import save_image
from utils import save_single_image

# path to python_utils
sys.path.insert(0, '../utils')
sys.path.insert(0, '/home/zenn')

from python_utils.LossWriter import LossWriter

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(configuration):
    """
    this is the main training loop
    :param configuration: the config
    :return:
    """
    image_saving_path = configuration['image_saving_path']
    print('saving result images to {}'.format(image_saving_path))

    model_dir = configuration['model_dir']
    print('vgg-19 model dir is {}'.format(model_dir))

    vgg_model = get_vgg_model(configuration)
    print('got vgg model')

    style_image_path = configuration['style_image_path']
    number_style_images, style_image_file_paths = get_images(style_image_path)
    print('got {} style images'.format(number_style_images))
    print('using the style images from path: {}'.format(style_image_path))

    content_image_path = configuration['content_image_path']
    number_content_images, content_image_file_paths = get_images(content_image_path)
    print('got {} content images'.format(number_content_images))
    print('using the content images from path: {}'.format(content_image_path))

    steps = configuration['steps']
    print('training for {} steps'.format(steps))

    content_weight = configuration['content_weight']
    style_weight = configuration['style_weight']
    print('content weight: {}, style weight: {}'.format(content_weight, style_weight))

    loss_writer = LossWriter(os.path.join(configuration['folder_structure'].get_parent_folder(), './loss/loss'))
    loss_writer.write_header(columns=['iteration', 'style_loss', 'content_loss', 'loss'])

    for i in range(number_style_images):
        print('style image {}'.format(i))
        for j in range(number_content_images):
            images = []
            print('content image {}'.format(j))
            style_image = load_image(style_image_file_paths[i])
            content_image = load_image(content_image_file_paths[j])

            images += [style_image.squeeze(0).cpu()]
            print('got style image')
            images += [content_image.squeeze(0).cpu()]
            print('got content image')

            for k in range(1, 6):
                print('training transfer image with loss {}'.format(k))
                torch.manual_seed(1)
                image_noise = torch.randn(style_image.data.size()).to(device)
                model, style_losses, content_losses = get_full_style_model(configuration,
                                                                           vgg_model,
                                                                           style_image,
                                                                           content_image,
                                                                           get_style_loss_module(k),
                                                                           get_content_loss_module())

                # this is to align the loss magnitudes of Gram matrix loss and moment loss
                if k == 1:
                    style_weight *= 100

                img = train_neural_style_transfer(model, style_losses, content_losses,
                                                  image_noise, steps, style_weight, content_weight, loss_writer).squeeze(0).cpu()

                images += [img.clone()]

                save_single_image(configuration, img, -k, -k)

                print('got transfer image')

            save_image(configuration, images, i, j)


def train_mmd(configuration):
    """
    this is the MMD training loop
    :param configuration: the config
    :return:
    """
    image_saving_path = configuration['image_saving_path']
    print('saving result images to {}'.format(image_saving_path))

    model_dir = configuration['model_dir']
    print('vgg-19 model dir is {}'.format(model_dir))

    vgg_model = get_vgg_model(configuration)
    print('got vgg model')

    style_image_path = configuration['style_image_path']
    number_style_images, style_image_file_paths = get_images(style_image_path)
    print('got {} style images'.format(number_style_images))
    print('using the style images from path: {}'.format(style_image_path))

    content_image_path = configuration['content_image_path']
    number_content_images, content_image_file_paths = get_images(content_image_path)
    print('got {} content images'.format(number_content_images))
    print('using the content images from path: {}'.format(content_image_path))

    loss_writer = LossWriter(os.path.join(configuration['folder_structure'].get_parent_folder(), './loss/loss'))
    loss_writer.write_header(columns=['iteration', 'style_loss', 'content_loss', 'loss'])

    print(style_image_file_paths)
    print(content_image_file_paths)

    images = []

    for i in range(number_style_images):
        print('style image {}'.format(i))
        for j in range(number_content_images):
            style_image = load_image(style_image_file_paths[i])
            content_image = load_image(content_image_file_paths[j])

            images += [style_image.squeeze(0).cpu()]
            print('got style image')
            images += [content_image.squeeze(0).cpu()]
            print('got content image')

            print('training transfer image with loss {} (MMD loss)'.format(2))
            torch.manual_seed(1)
            image_noise = torch.randn(style_image.data.size()).to(device)
            model, style_losses, content_losses = get_full_style_model(configuration,
                                                                       vgg_model,
                                                                       style_image,
                                                                       content_image,
                                                                       get_style_loss_module(2),
                                                                       get_content_loss_module())

            steps = configuration['steps']
            print('training for {} steps'.format(steps))

            content_weight = configuration['content_weight']
            style_weight = configuration['style_weight']
            print('content weight: {}, style weight: {}'.format(content_weight, style_weight))

            img = train_neural_style_transfer(model, style_losses, content_losses,
                                              image_noise, steps, style_weight, content_weight, loss_writer).squeeze(0).cpu()

            save_image(configuration, img, j, i)

            print('got transfer image')


def train_neural_style_transfer(model, style_losses, content_losses, image_noise, steps, style_weight, content_weight,
                                loss_writer):
    """
    the actual training of the model
    :param model: the pre-trained VGG-19 model
    :param style_losses: a list of style losses to be inserted to the model
    :param content_losses: a list of content losses to be inserted to the model
    :param image_noise: the noise image
    :param steps: the number of steps to train
    :param style_weight: the weighting factor for the style loss term
    :param content_weight: the weighting factor for the content loss term
    :param loss_writer: loss writer that writes the loss to csv
    :return: the stylized image
    """
    optimizer = get_input_optimizer(image_noise)
    model.to(device)

    print('style weight: {}, content weight: {}'.format(style_weight, content_weight))
    print('Optimizing.. {} steps'.format(steps))
    run = [0]
    while run[0] <= steps:
        def closure():
            # correct the values of updated input image
            image_noise.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(image_noise)

            style_score = 0
            content_score = 0

            style_loss_factor = 1 / len(style_losses)
            content_loss_factor = 1 / len(content_losses)

            for sl in style_losses:
                style_score += style_loss_factor * sl.loss
            for cl in content_losses:
                content_score += content_loss_factor * cl.loss

            content_loss = content_weight * content_score
            style_loss = style_weight * style_score

            loss = style_loss + content_loss

            loss.to(device)
            loss.backward(retain_graph=True)

            # print('style loss: {:4f}, content loss: {:4f}'.format(style_score.item(), content_score.item()))

            if run[0] % 10000 == 0:
                print("run {} and {} to go".format(run[0], steps - run[0]))
                print('style loss: {:4f}, content loss: {:4f}'.format(style_score.item(), content_score.item()))
                print('loss: {:4f}'.format(loss.item()))
                print()

            run[0] += 1

            return style_loss.item(), content_loss.item(), loss.item()

        style_loss, content_loss, loss = optimizer.step(closure)

        loss_writer.write_row([run[0], style_loss, content_loss, loss])

    # transform image to [0, 1]
    image_noise.data.clamp_(0, 1)

    return image_noise
