import os
import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, roc_curve
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags, logging
from data import DataCIFAR10, DataGTSRB
from model import Model, ModelSAP, SimpleModel

torch.manual_seed(23)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'dataset',
    default='cifar10',
    enum_values=['cifar10', 'GTSRB_processed'],
    help="Dataset for experiments."
)

flags.DEFINE_bool(
    'is_train',
    default=False,
    help="If this flag is set, train a model with --train_method."
)

flags.DEFINE_enum(
    'model',
    default='simple',
    enum_values=['simple', 'normal', 'normalSAP'],
    help="Model."
)

flags.DEFINE_enum(
    'train_method',
    default='none',
    enum_values=['none', 'fgsm', 'rfgsm', 'ifgsm', 'mifgsm'],
    help="Method of adversarial attack."
)

flags.DEFINE_integer(
    'epochs',
    default=150,
    help="Training epochs."
)

flags.DEFINE_integer(
    'batch_size',
    default=32,
    help="Batch size."
)

flags.DEFINE_float(
    'epsilon',
    default=4. / 255,
    help="Scale of adversarial attacks, meaning [x - epsilon, x + epsilon]"
)

flags.DEFINE_float(
    'alpha',
    default=2. / 255,
    help="Noise addition fraction in RFGSM."
)

flags.DEFINE_integer(
    'step',
    default=20,
    help="Iterative number for IFGSM or MIFGSM."
)

flags.DEFINE_bool(
    'use_atda_loss',
    default=False,
    help="If this flag is set, train a model with ATDA and --train_method."
)

flags.DEFINE_bool(
    'is_test',
    default=False,
    help="If this flag is set, test a trained moedel with --test_method."
)

flags.DEFINE_string(
    'model_name_for_test',
    default=None,
    help="Model name that will be tested like 'model_none_simple(.pt|)'."
)

flags.DEFINE_enum(
    'test_method',
    default='none',
    enum_values=['none', 'fgsm', 'rfgsm', 'ifgsm', 'mifgsm'],
    help="Method of adversarial attack."
)

flags.DEFINE_bool(
    'is_kde_test',
    default=False,
    help="If this flag is set, test a trained moedel with KDE."
)

flags.DEFINE_bool(
    'is_random_crop_test',
    default=False,
    help="If this flag is set, test a trained moedel with random resize and crop."
)


class AdvParams():
    """AdvParams represents parameters of adversarial attacks/defences.

    Attributes
    ----------
    self.sampler torch.distribution : noise sampler for rfgsm (fixed as Normal[-1,1]).
    self.epsilon float : scale of adversary [x - epsilon, x + epsilon].
    self.alpha float : scale of noisze [x - alpha, x + alpha].
    self.step int : iteration numbers for ifgsm/mifgsm.
    self.train_method str : specify train adversarial method such "fgsm".
    self.test_method str : specify test adversarial method such "fgsm".
    self.is_train bool : indicate training/test.
    """
    def __init__(self, epsilon, alpha, step, train_method, test_method, is_train=True):
        self.sampler = torch.distributions.Normal(-1.0, 1.0)
        self.epsilon = epsilon
        self.alpha = alpha
        self.step = step
        self.train_method = train_method
        self.test_method = test_method
        self.is_train = is_train


def train(model, trainloader, save_path, epoch):
    """Train and save a model.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    trainloader torch.utils.data.DataLoader : train data loader.
    save_path str : model save path.
    epoch int : epochs for model training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, epoch + 1):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 0:
                logging.info(f"[{epoch}, {i:>5}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

    logging.info('Finished Training.')
    torch.save(model.state_dict(), save_path)


def _load_model_weight(model, model_path):
    """Load trained weights into a model.

    Model case - just load trained weights.
    ModelSAP case - skip SAP layers to load trained weights appropriately.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    model_path str : path to model trained in advance.

    Returns
    -------
    model Model(nn.Module) : PyTorch model with loaded weights.
    """
    if type(model) == SimpleModel or type(model) == Model:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    elif type(model) == ModelSAP:
        pretrained_model = Model(model.num_classes).to(device)
        pretrained_model.load_state_dict(torch.load(model_path,
                                         map_location=torch.device(device)))
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

    return model


def test(model, testloader, model_path):
    """Test a trained model.

    Results will be written in a log file.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    testloader torch.utils.data.DataLoader : test data loader.
    model_path str : path to a traiend model.
    """
    model = _load_model_weight(model, model_path)
    model.eval()

    correct = 0
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        final_pred = output.max(1, keepdim=True)[1]  # [1] : indices.
        correct += int(sum(final_pred.flatten() == y))

    final_acc = correct / float(testloader.__len__() * testloader.batch_size)
    logging.info(f"Accuracy on test data: {final_acc}")


def _gen_grad(x, y, logit, model, is_train):
    """Generate loss gradients of data.

    If is_train is True, use model predictions as labels of loss function
    to avoid label leaking (https://arxiv.org/abs/1611.01236).

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    logit torch.Tensor : logit tensor whose shape is [b, num_classes].
    model Model(nn.Module) : Pytorch model.
    is_train bool : Flag to denote training or not.

    Returns
    -------
    data_grad torch.Tensor : loss gradients of data whose shape is [b, c, h, w].
    """
    x.retain_grad()
    if is_train:
        y_model = logit.max(1, keepdim=False)[1].long().to(device)  # [1] : indices.
        loss = F.nll_loss(F.log_softmax(logit, dim=1), y_model)
    else:
        loss = F.nll_loss(F.log_softmax(logit, dim=1), y)
    model.zero_grad()
    loss.backward(retain_graph=True)
    data_grad = x.grad.data

    return data_grad


def fgsm_attack(x, y, logit, model, adv_params):
    """Generate loss gradients of data.

    If is_train is True, use model predictions as labels of loss function
    to avoid label leaking (https://arxiv.org/abs/1611.01236).

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    logit torch.Tensor : logit tensor whose shape is [b, num_classes].
    model Model(nn.Module) : PyTorch model.
    adv_params AdvParams : parameters of adversary.

    Returns
    -------
    x_adv torch.Tensor : perturbated x whose shape is [b, c, h, w].
    """
    data_grad = _gen_grad(x, y, logit, model, adv_params.is_train)
    sign_data_grad = data_grad.sign()
    x_adv = x + adv_params.epsilon * sign_data_grad
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


def rfgsm_attack(x, y, logit, model, adv_params):
    """Generate loss gradients of data.

    Randomized FGSM: https://arxiv.org/abs/1705.07204.

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    logit torch.Tensor : logit tensor whose shape is [b, num_classes].
    model Model(nn.Module) : PyTorch model.
    adv_params AdvParams : parameters of adversary.

    Returns
    -------
    x_adv torch.Tensor : perturbated x whose shape is [b, c, h, w].
    """
    sign_noise = adv_params.sampler.sample(x.shape).to(device).sign()
    x_noise = x + adv_params.alpha * sign_noise
    x_noise = torch.clamp(x_noise, 0, 1)
    logit_noise = model(x_noise)
    adv_params.epsilon = adv_params.epsilon - adv_params.alpha
    x_adv = fgsm_attack(x_noise, y, logit_noise, model, adv_params)
    adv_params.epsilon = adv_params.epsilon + adv_params.alpha

    return x_adv


def ifgsm_attack(x, y, model, adv_params):
    """Generate loss gradients of data.

    Projected Gradient Descent : https://arxiv.org/abs/1706.06083.

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    model Model(nn.Module) : PyTorch model.
    adv_params AdvParams : parameters of adversary.

    Returns
    -------
    x_adv torch.Tensor : perturbated x whose shape is [b, c, h, w].
    """
    x_adv = x

    epsilon_org = adv_params.epsilon
    adv_params.epsilon = epsilon_org / 10.0
    for _ in range(adv_params.step):
        logit = model(x_adv)
        x_adv = fgsm_attack(x_adv, y, logit, model, adv_params)
        # Clip x_adv within [x - eps, x + eps]
        x_adv = torch.max(torch.min(x_adv, x + epsilon_org), x - epsilon_org)
        x_adv = torch.clamp(x_adv, 0, 1)

    adv_params.epsilon = epsilon_org
    return x_adv


def mifgsm_attack(x, y, model, adv_params):
    """Generate loss gradients of data.

    Momentum Iterative FGSM: https://arxiv.org/abs/1710.06081.

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    model Model(nn.Module) : PyTorch model.
    adv_params AdvParams : parameters of adversary.

    Returns
    -------
    x_adv torch.Tensor : perturbated x whose shape is [b, c, h, w].
    """
    decay_factor = 1.0
    scale = adv_params.epsilon / 5.0

    momentum = torch.zeros_like(x)
    x_adv = x

    for _ in range(adv_params.step):
        outputs = model(x_adv)
        data_grad = _gen_grad(x_adv, y, outputs, model, adv_params.is_train)
        reduce_idx = list(range(1, len(data_grad.shape)))
        denominator = torch.mean(torch.abs(data_grad), reduce_idx, keepdim=True)
        data_grad = data_grad / torch.max(denominator, denominator + 1e-12)
        momentum = decay_factor * momentum + data_grad

        sign_momentum = data_grad.sign()
        scaled_grad = scale * sign_momentum
        x_adv = x_adv + scaled_grad
        # Clip x_adv within [x - eps, x + eps]
        x_adv = torch.max(torch.min(x_adv, x + adv_params.epsilon),
                          x - adv_params.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


def create_adv_sample(x, y, logit, model, adv_params, adv_method):
    """Create adversarial examples using adv_method.

    Parameters
    ----------
    x torch.Tensor : input data whose shape is [b, c, h, w].
    y torch.Tensor : true label whose shape is [b].
    logit torch.Tensor : logit tensor whose shape is [b, num_classes].
    model Model(nn.Module) : PyTorch model.
    adv_params AdvParams : parameters of adversary.
    adv_method str : adversary method used to create x_adv.

    Returns
    -------
    x_adv torch.Tensor : perturbated x whose shape is [b, c, h, w].
    """
    if adv_method == "fgsm":
        x_adv = fgsm_attack(x, y, logit, model, adv_params)
    elif adv_method == "rfgsm":
        x_adv = rfgsm_attack(x, y, logit, model, adv_params)
    elif adv_method == "ifgsm":
        x_adv = ifgsm_attack(x, y, model, adv_params)
    elif adv_method == "mifgsm":
        x_adv = mifgsm_attack(x, y, model, adv_params)

    return x_adv


def train_adv(model, trainloader, save_path, epoch, adv_params):
    """Train a model using adversarial training and save the trained model.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    trainloader torch.utils.data.DataLoader : train DataLoader
    save_path str : model save path.
    epoch int : epochs for model training.
    adv_params AdvParams : parameters of adversary.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, epoch + 1):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            x.requires_grad = True

            logit = model(x)
            x_adv = create_adv_sample(x, y, logit, model, adv_params,
                                      adv_params.train_method)
            logit_adv = model(x_adv)

            # Adversarial training
            optimizer.zero_grad()
            adv_loss = (0.8 * F.cross_entropy(logit, y)
                        + (1 - 0.8) * F.cross_entropy(logit_adv, y))
            adv_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += adv_loss.item()
            if i % 200 == 0:
                logging.info(f"[{epoch}, {i:>5}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

    logging.info('Finished Training.')
    torch.save(model.state_dict(), save_path)


def train_atda(model, trainloader, save_path, epoch, adv_params):
    """Train a model using adversarial training with domain adaptation.

    This training method is based on https://arxiv.org/abs/1810.00740.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    trainloader torch.utils.data.DataLoader : train DataLoader
    save_path str : model save path.
    epoch int : epochs for model training.
    adv_params AdvParams : parameters of adversary.
    """
    def _coral_loss(source, target):
        """Compute CORAL loss between source and target.

        Parameters
        ----------
        source torch.Tensor : tensor of source domain.
        target torch.Tensor : tensor of target domain.

        Returns
        -------
        loss torch.Tensor : loss tensor.
        """
        mean_s = torch.mean(source, dim=0, keepdim=True) - source
        covariance_s = torch.matmul(torch.transpose(mean_s, 0, 1), mean_s)
        mean_t = torch.mean(target, dim=0, keepdim=True) - target
        covariance_t = torch.matmul(torch.transpose(mean_t, 0, 1), mean_t)
        loss = torch.mean(torch.abs(covariance_s - covariance_t))
        return loss

    def _mmd_loss(source, target):
        """Compute MMD loss between source and target.

        Parameters
        ----------
        source torch.Tensor : tensor of source domain.
        target torch.Tensor : tensor of target domain.

        Returns
        -------
        loss torch.Tensor : loss tensor.
        """
        mean_s = torch.mean(source, dim=0) - source
        mean_t = torch.mean(target, dim=0) - target
        loss = torch.mean(torch.abs(mean_s - mean_t))
        return loss

    def _margin_loss(logit, logit_adv, y, centers):
        """Compute margin loss between source and target.

        Parameters
        ----------
        logit torch.Tensor : logit tensor of source domain.
        logit torch.Tensor : logit tensor of target domain.
        y torch.Tensor : label.
        centers torch.Tensor : class centers in the logit space.

        Returns
        -------
        loss torch.Tensor : loss tensor.
        centers torch.Tensor : updated class centers.
        """
        # Parameter
        alpha = 0.1

        concat_logit = torch.cat((logit, logit_adv), dim=0)  # [2 * b, len(logit)]
        concat_y = torch.cat((y, y), dim=0)  # [2 * b]
        centers_batch = centers[concat_y, :]  # [2 * b, len(logit)]
        centers_dist = torch.mean(
            torch.abs(concat_logit - centers_batch), dim=1)  # [2 * b]

        diff_batch = centers_batch - concat_logit
        unique_num, unique_idx, unique_count = torch.unique(concat_y,
                                                            return_inverse=True,
                                                            return_counts=True)
        appearance_num = unique_count[unique_idx].unsqueeze(1).float()  # [2 * b, 1]
        diff_batch = alpha * (diff_batch / (1. + appearance_num))
        diff = torch.zeros_like(centers).index_add_(
            0, concat_y, diff_batch)  # [num_classes, len(logit)]

        # Update center positions.
        centers = centers - diff.data

        logit_center_pair_dist = torch.sum(
            torch.abs(concat_logit.unsqueeze(1) - centers.unsqueeze(0)),
            # [2 * b, num_classes, len(logit)]
            dim=2)  # [2 * b, num_classes]
        logit_center_dist = centers_dist.unsqueeze(1) - logit_center_pair_dist
        # logit_center_dist: [2 * b, num_classes]
        logit_center_labels_equal = (concat_y.unsqueeze(1) == torch.Tensor(
            [c for c in range(centers.shape[0])]).to(device).unsqueeze(0))
        # logit_center_labels_equal: [2 * b, num_classes]
        mask = torch.logical_not(logit_center_labels_equal)

        loss = torch.sum(F.softplus(logit_center_dist) * mask) / torch.sum(mask)

        return loss, centers

    centers = torch.zeros([model.num_classes, model.num_classes],
                          dtype=torch.float32, requires_grad=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, epoch + 1):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            x.requires_grad = True

            logit = model(x)
            x_adv = create_adv_sample(x, y, logit, model, adv_params,
                                      adv_params.train_method)
            logit_adv = model(x_adv)

            # Adversarial training with domain adaptation
            optimizer.zero_grad()
            coral_loss = _coral_loss(logit, logit_adv)
            mmd_loss = _mmd_loss(logit, logit_adv)
            margin_loss, centers = _margin_loss(logit, logit_adv, y, centers)
            adv_loss = (F.cross_entropy(logit, y)
                        + F.cross_entropy(logit_adv, y)
                        + 1 / 3. * (coral_loss + mmd_loss + margin_loss))
            adv_loss.backward()
            optimizer.step()

            # Logging
            running_loss += adv_loss.item()
            if i % 200 == 0:
                logging.info(f"[{epoch}, {i:>5}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

    logging.info('Finished Training.')
    torch.save(model.state_dict(), save_path)


def test_adv(model, testloader, model_path, adv_params, adv_img_save_base):
    """Test a trained mode with adversarial test data.

    Results will be written in a log file.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    testloader torch.utils.data.DataLoader : test data loader.
    model_path str : path to a traiend model.
    adv_params AdvParams : parameters of adversary.
    adv_img_save_base str : base path to save adversarial examples.
    """
    # Parameter
    save_adv_img_num = 5

    model = _load_model_weight(model, model_path)
    model.eval()

    correct = 0
    misclassified_adv_examples = []

    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True

        logit = model(x)
        init_pred = logit.max(1, keepdim=True)[1].flatten()  # [1] : indices.
        x_adv = create_adv_sample(x, y, logit, model, adv_params,
                                  adv_params.test_method)
        logit_adv = model(x_adv)

        final_pred = logit_adv.max(1, keepdim=True)[1].flatten()  # [1] : indices.
        for x_i, x_adv_i, y_i, ip_i, fp_i in zip(x, x_adv, y, init_pred, final_pred):
            if fp_i == y_i:
                correct += 1
            elif ip_i == y_i and len(misclassified_adv_examples) < save_adv_img_num:
                misclassified_adv_examples.append((x_i, x_adv_i))

    final_acc = correct / float(testloader.__len__() * testloader.batch_size)
    logging.info(f"Accuracy on test_adv data: {final_acc}")

    # Save adversarial examples
    for idx, (x_i, x_adv_i) in enumerate(misclassified_adv_examples, start=1):
        x_i_np = x_i.transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        x_adv_i_np = x_adv_i.transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        im = Image.fromarray((x_i_np * 255).astype(np.uint8))
        im_adv = Image.fromarray((x_adv_i_np * 255).astype(np.uint8))
        im_merged = Image.new('RGB', (2 * im.width, im.height))
        im_merged.paste(im, (0, 0))
        im_merged.paste(im_adv, (im.width, 0))
        im_merged.save(f"{adv_img_save_base}{idx}.png")


def exp_kde(model, trainloader, testloader, model_path, adv_params):
    """Test a trained mode with adversarial test data.

    Results will be written in a log file.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    trainloader torch.utils.data.DataLoader : train data loader.
    testloader torch.utils.data.DataLoader : test data loader.
    model_path str : path to a traiend model.
    adv_params AdvParams : parameters of adversary.
    """
    def _compute_logits(loader, num_classes, adv_method):
        """Compute logits and prediction labels of model.

        Parameters
        ----------
        loader torch.utils.data.DataLoader : data loader of train/test.
        num_classes int : number of classes.
        adv_method str : adversary method to compute logit.

        Returns
        -------
        logits torch.Tensor : logits obtained from model.
        labels torch.Tensor : labels.
        labels_pred torch.Tensor : model prediction labels.
        """
        logits = np.zeros(shape=(len(loader.dataset), num_classes))
        labels = np.zeros(shape=(len(loader.dataset)))
        labels_pred = np.zeros(shape=(len(loader.dataset)))
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            x.requires_grad = True

            logit = model(x)

            if adv_method == "none":
                pass
            else:
                x_adv = create_adv_sample(x, y, logit, model, adv_params, adv_method)
                # Overwrite logit.
                logit = model(x_adv)

            pred = logit.max(1, keepdim=True)[1].flatten()  # [1] : indices.
            start = idx * loader.batch_size
            end = (idx + 1) * loader.batch_size
            logits[start : end, :] = logit.cpu().detach().numpy()
            labels[start : end] = y.cpu().detach().numpy().astype(np.int)
            labels_pred[start : end] = pred.cpu().detach().numpy().astype(np.int)

        return logits, labels, labels_pred

    def _compute_kde_score(label, feature):
        """Compute Kernel Density Estimation of feature by kdes[label].

        Parameters
        ----------
        label torch.Tensor (single data) : label.
        feature torch.Tensor (single data) : feature.

        Returns
        -------
        float : computed kde score.
        """
        return float(kdes[label].score_samples(np.reshape(feature, (1, -1))).squeeze())

    def _compute_densities(labels, features):
        """Compute KDE densities.

        Parameters
        ----------
        labels torch.Tensor : labels.
        features torch.Tensor : features.

        Returns
        -------
        densities List[float] : all densities.
        """
        densities = []
        for label, feature in zip(labels, features):
            densities.append(_compute_kde_score(label, feature))

        return densities

    model = _load_model_weight(model, model_path)
    model.eval()

    logging.info(f"Train KDEs for each class.")

    logits_train, labels_train, _ = _compute_logits(trainloader, model.num_classes,
                                                    "none")

    kdes = {}
    for class_idx in range(model.num_classes):
        kdes[class_idx] = KernelDensity(kernel='gaussian', bandwidth=2.25).fit(
            logits_train[np.where(labels_train == class_idx)])

    logging.info(f"Finished the training.")

    logging.info(f"Compute densities for both clean and adv. test data.")

    # Clean test data.
    logits_test, labels_test, labels_pred_test = _compute_logits(testloader,
                                                                 model.num_classes,
                                                                 "none")
    densities = _compute_densities(labels_pred_test, logits_test)

    # Adversarial test data.
    logits_test_adv, _, labels_pred_adv = _compute_logits(testloader, model.num_classes,
                                                          adv_params.test_method)
    densities_adv = _compute_densities(labels_pred_adv, logits_test_adv)

    logging.info(f"Finished computing the densities.")

    logging.info(f"Evaluate the computed densities")

    # d is log(prob), so p(x_adv) / p(x) < 1 is d(x) / d(x_adv) < 1
    ratios = [d / d_adv for (d, d_adv) in zip(densities, densities_adv)]
    ratios_smaller_than_one = sum(map(lambda x: x < 1, ratios)) / len(ratios)
    logging.info(f"Result (p(x_adv) / p(x) < 1): {ratios_smaller_than_one}.")

    features = np.reshape(np.concatenate([densities, densities_adv]), (-1, 1))
    labels = np.concatenate([np.zeros_like(densities), np.ones_like(densities_adv)])
    lr = LogisticRegressionCV(n_jobs=-1, random_state=23).fit(features, labels)
    accuracy = sum(lr.predict(features) == labels) / len(labels)
    logging.info(f"Result (ACC): {accuracy}.")

    probs = lr.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    logging.info(f"Result (ROC-AUC): {auc_score}.")

    logging.info(f"Finished evaluating the densities.")


def exp_random_crop(model, testloader, model_path, adv_params, size):
    """Test a trained model with random resize and crop.

    Results will be written in a log file.

    Parameters
    ----------
    model Model(nn.Module) : PyTorch model such as Model.
    testloader torch.utils.data.DataLoader : test data loader.
    model_path str : path to a traiend model.
    adv_params AdvParams : parameters of adversary.
    size int : original image will be scaled to this size.
    """
    def _random_resize_crop(img, size):
        img = img.unsqueeze(0)
        # Randomly resize the image.
        resize = random.randint(img.shape[-1], size - 1)
        resized_img = F.interpolate(img, (resize, resize))
        # 0-pad the resized image. 0-pad to all left, right, top and bottom.
        pad_size = size - resize
        padded_img = F.pad(resized_img, (pad_size,) * 4)
        # Crop the padded image to get (size, size) image.
        pos_top = random.randint(0, pad_size)
        pos_left = random.randint(0, pad_size)
        return padded_img[0, :, pos_top:pos_top + size, pos_left:pos_left + size]

    model = _load_model_weight(model, model_path)
    model.eval()

    correct = 0
    correct_adv = 0

    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True

        batches, channels, _, _ = x.shape
        transformed_x = torch.zeros([batches, channels, size, size]).to(device)
        for b in range(batches):
            transformed_img = _random_resize_crop(x[b, :, :, :], size)
            transformed_x[b, :, :, :] = transformed_img
        logit = model(transformed_x)
        pred = logit.max(1, keepdim=True)[1].flatten()  # [1] : indices.

        x_adv = create_adv_sample(x, y, logit, model, adv_params,
                                  adv_params.test_method)
        transformed_x_adv = torch.zeros([batches, channels, size, size]).to(device)
        for b in range(batches):
            transformed_img_adv = _random_resize_crop(x_adv[b, :, :, :], size)
            transformed_x_adv[b, :, :, :] = transformed_img_adv
        logit_adv = model(transformed_x_adv)
        pred_adv = logit_adv.max(1, keepdim=True)[1].flatten()  # [1] : indices.

        for y_i, p_i, p_adv_i in zip(y, pred, pred_adv):
            if y_i == p_i:
                correct += 1
            if y_i == p_adv_i:
                correct_adv += 1

    final_acc = correct / float(testloader.__len__() * testloader.batch_size)
    final_acc_adv = correct_adv / float(testloader.__len__() * testloader.batch_size)
    logging.info(f"Accuracy on test data: {final_acc}")
    logging.info(f"Accuracy on test_adv data: {final_acc_adv}")


def main(argv):
    def _loggig_all_flags():
        """Logging information of all flags.
        """
        for k, v in FLAGS.__flags.items():
            logging.info(f"{k} : {v.value}")

    def _generate_model_path(is_train=True, model_name=None):
        """Generate model path using FLAGS information.

        Parameters
        ----------
        is_train bool : train or test.
        model_name str : model name to be loaded in the test phase.

        Returns
        -------
        model_path str : model path to save/load a model.
        """
        model_dir = f"./model/{FLAGS.dataset}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/"

        if is_train:
            model_path += f"model_{FLAGS.model}_{FLAGS.train_method}"
            if FLAGS.use_atda_loss:
                model_path += "_atda"
            model_path += ".pt"
        else:
            model_path += model_name.split(".pt")[0]
            model_path += ".pt"

        return model_path

    logging.set_verbosity(logging.INFO)
    if FLAGS.log_dir != '':
        logging.get_absl_handler().use_absl_log_file()
    _loggig_all_flags()

    if FLAGS.dataset == "cifar10":
        Data = DataCIFAR10(batch_size=FLAGS.batch_size)
    elif FLAGS.dataset == "GTSRB_processed":
        Data = DataGTSRB(batch_size=FLAGS.batch_size)
    trainloader, testloader, num_classes = Data.prepare_data()

    model_path = _generate_model_path()

    if FLAGS.model == "simple":
        model = SimpleModel(num_classes).to(device)
    elif FLAGS.model == "normal":
        model = Model(num_classes).to(device)
    elif FLAGS.model == "normalSAP":
        model = ModelSAP(num_classes).to(device)

    adv_params = AdvParams(FLAGS.epsilon, FLAGS.alpha, FLAGS.step,
                           FLAGS.train_method, FLAGS.test_method)
    if FLAGS.is_train:
        if FLAGS.train_method == "none":
            train(model, trainloader, model_path, FLAGS.epochs)
        else:
            adv_params.is_train = True
            if FLAGS.use_atda_loss:
                train_atda(model, trainloader, model_path, FLAGS.epochs, adv_params)
            else:
                train_adv(model, trainloader, model_path, FLAGS.epochs, adv_params)

    if FLAGS.model_name_for_test is not None:
        model_path = _generate_model_path(False, FLAGS.model_name_for_test)

    if FLAGS.is_test:
        if FLAGS.test_method == "none":
            test(model, testloader, model_path)
        else:
            adv_params.is_train = False
            adv_img_save_base = "./data/adv_img_"
            adv_img_save_base += f"{FLAGS.dataset}_{FLAGS.model}_"
            adv_img_save_base += f"{FLAGS.train_method}_{FLAGS.test_method}_"
            test_adv(model, testloader, model_path, adv_params, adv_img_save_base)

    if FLAGS.is_kde_test:
        exp_kde(model, trainloader, testloader, model_path, adv_params)

    if FLAGS.is_random_crop_test:
        adv_params.is_train = False
        if FLAGS.dataset == "cifar10":
            exp_random_crop(model, testloader, model_path, adv_params, 34)
        elif FLAGS.dataset == "GTSRB_processed":
            exp_random_crop(model, testloader, model_path, adv_params, 55)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
