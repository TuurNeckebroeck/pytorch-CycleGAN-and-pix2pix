import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


class ConditionalCycleGAN2Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    @staticmethod
    def onehot_encode_colors(color_list, emb_shape=(256,256), nb_labels=12):
        assert len(emb_shape) == 2
        assert color_list.min().item() >= 0 and color_list.max().item() < nb_labels

        encoding = torch.zeros((len(color_list), nb_labels, *emb_shape))
        for idx, color in enumerate(color_list):
            if isinstance(color, torch.Tensor):
                color = int(color.item())
            encoding[idx,color,:,:] = torch.ones(emb_shape)
        return encoding


    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.loss_names = ['D_img_A', 'D_color_A', 'D_img_B', 'D_color_B', 'G_A', 'cycle_A', 'idt_A', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D_img', 'D_color']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        nb_color_channels = 12

        self.netG = networks.define_G(opt.input_nc + nb_color_channels, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define discriminators
            self.netD_img = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_color = networks.define_D(opt.output_nc + nb_color_channels, opt.ndf, opt.netD,
            #                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_color = networks.define_D_color(opt.init_type, opt.init_gain, self.gpu_ids)
            # resnet per definitie 3 inputchannels

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_pool= ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionColor = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # removed itertools.chain
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # removed itertools.chain
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_img.parameters(),self.netD_color.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.color_A = input['color_A'].to(self.device)

        self.real_B = input['B'].to(self.device)
        self.color_B = input['color_B'].to(self.device)

        # TODO wat hiermee doen?
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        """forward functie blijft onveranderd"""
        # self.fake_B = self.netG(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG(self.fake_A)   # G_A(G_B(B))


        # concat input image with color to generate
        color_B = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_B).cuda()
        self.fake_B = self.netG(torch.cat([self.real_A, color_B], dim=1))  # G(A)
        color_A = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_A).cuda()
        self.rec_A = self.netG(torch.cat([self.fake_B, color_A], dim=1))   # G(G(A))

        self.fake_A = self.netG(torch.cat([self.real_B, color_A], dim=1))  # G(A)
        self.rec_B = self.netG(torch.cat([self.fake_A, color_B], dim=1))   # G(G(B))

    def backward_D_basic(self, netD_img, netD_color, real, colors_real, fake, colors_fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        nb_images_real = real.shape[0]
        img_shape = real.shape[2:4]
        nb_colors = 12
        assert len(colors_real) == nb_images_real
        # colors_real_encoded = ConditionalCycleGAN2Model.onehot_encode_colors(colors_real.detach(), emb_shape=img_shape).cuda()
        # colors_fake_encoded = ConditionalCycleGAN2Model.onehot_encode_colors(colors_fake.detach(), emb_shape=img_shape).cuda()
        colors_real_encoded = torch.nn.functional.one_hot(colors_real.detach(), num_classes=nb_colors).cuda()
        # colors_fake_encoded = torch.nn.functional.one_hot(colors_fake.detach(), nb_classes=12).cuda() # niet nodig volgens formulering ACGAN

        # Real
        # d_img_input_real = torch.cat([real.detach(), colors_real_encoded], dim=1)
        # pred_img_real = netD_img(d_img_input_real.detach())
        loss_D_img_real = self.criterionGAN(netD_img(real.detach()), True)
        
        # Fake image + correct color
        # d_input_fake_img = torch.cat([fake.detach(), colors_fake_encoded], dim=1)
        # pred_fake_img = netD_img(d_input_fake_img.detach())
        loss_D_img_fake = self.criterionGAN(netD_img(fake.detach()), False)

        colors_real_predicted = netD_color(real.detach())
        # loss_D_color_real = self.criterionColor(colors_real_predicted, colors_real_encoded.detach())
        loss_D_color_real = self.criterionColor(colors_real_predicted, colors_real.detach())
        colors_fake_predicted = netD_color(fake.detach())
        loss_D_color_fake = self.criterionColor(colors_fake_predicted, colors_real.detach())

        # Real image + incorrect color
        # calculate random fake colors
        # TODO replace hardcoded number of colors
        # colors_incorrect = np.random.randint(0, nb_colors, nb_images_real)
        # idxes_same = np.where(colors_incorrect == colors_fake)
        # colors_incorrect[idxes_same] = (colors_incorrect[idxes_same] + 1) % nb_colors
        # colors_incorrect = ConditionalCycleGAN2Model.onehot_encode_colors(colors_incorrect).cuda()

        # d_input_fake_label = torch.cat([real.detach(), colors_incorrect], dim=1)
        # pred_fake_label = netD(d_input_fake_label)
        # loss_D_fake_label = self.criterionGAN(pred_fake_label, False)

        # # Combined loss and calculate gradients
        # loss_D = loss_D_real + 0.5 * (loss_D_fake_label + loss_D_fake_img)
        # loss_D.backward() 
        # return loss_D
        loss_D = loss_D_img_real + loss_D_img_fake + loss_D_color_real + loss_D_color_fake
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        # TODO kleur querien?
        fake_B, (color_B_orig, color_B_new) = self.fake_pool.query(self.fake_B, attributes=torch.Tensor([(self.color_A, self.color_B)]))
        fake_A, (color_A_orig, color_A_new) = self.fake_pool.query(self.fake_A, attributes=torch.Tensor([(self.color_B, self.color_A)]))
        fake = torch.cat([fake_B, fake_A], dim=0)
        colors_fake = torch.cat([color_B_new.view(1), color_A_new.view(1)])
        real = torch.cat([self.real_A, self.real_B], dim=0)
        colors_real = torch.cat([self.color_A.view(1), self.color_B.view(1)], dim=0)
        self.loss_D = self.backward_D_basic(self.netD_img, self.netD_color, real, colors_real, fake, colors_fake)


    def backward_G(self):
        # TODO voor alle losses wordt resultaat opnieuw berekend terwijl dit opgeslaan wordt in def forward
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.idt_A = self.netG(self.real_B)
            # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            color_A = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_A).cuda()
            self.idt_A = self.netG(torch.cat([self.real_B, color_A], dim=1))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.idt_B = self.netG_B(self.real_A)
            # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            color_B = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_B).cuda()
            self.idt_B = self.netG(torch.cat([self.real_A, color_B], dim=1))
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD(self.fake_B), True)
        # color_B = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_B).cuda()
        colors_B_encoded = torch.nn.functional.one_hot(self.color_B.detach(), num_classes=12).cuda()
        self.loss_G_A_img = self.criterionGAN(self.netD_img(self.fake_B), True)
        self.loss_G_A_color = self.criterionColor(self.netD_color(self.fake_B), self.color_B.detach())

        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # color_A = ConditionalCycleGAN2Model.onehot_encode_colors(self.color_A).cuda()
        # self.loss_G_B = self.criterionGAN(self.netD(torch.cat([self.fake_A, color_A], dim=1)), True)
        colors_A_encoded = torch.nn.functional.one_hot(self.color_A.detach(), num_classes=12).cuda()
        self.loss_G_B_img = self.criterionGAN(self.netD_img(self.fake_A), True)
        self.loss_G_B_color = self.criterionColor(self.netD_color(self.fake_A), self.color_A.detach())
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A_img + self.loss_G_A_color + self.loss_G_B_img + self.loss_G_B_color + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_color, self.netD_img], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_color, self.netD_img], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
