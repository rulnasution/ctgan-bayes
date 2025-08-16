"""CTGAN module."""

import os,sys

current = os.path.dirname(os.path.realpath('__file__'))
parent = os.path.dirname(current)
sys.path.append(current+'/code/')

import warnings

import numpy as np
import pandas as pd
import torch
import itertools
import math
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, CrossEntropyLoss, Sigmoid
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from copy import deepcopy

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

from swag.utils import schedule, adjust_learning_rate, update_ema, bn_update
from swag.posteriors import SWAG
from swag.posteriors.diag_laplace import Laplace

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class log_variance(torch.nn.Module):
    def __init__(self, noise=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = Parameter(torch.DoubleTensor(1, 1))

        torch.nn.init.constant_(self.log_var, val=np.log(noise))

    def forward(self, x):
        return self.log_var * torch.ones(x.size(0),1).to(torch.device('cuda:0'))

def get_network_weights(model) -> tuple: 
    """
    Extract current network weight values as `np.ndarray`.

    :return: Tuple containing current network weight values
    ## adapted from https://github.com/automl/pybnn/blob/master/pybnn/bohamiann.py
    """
    return tuple(
        np.asarray(parameter.data.clone().detach().cpu().numpy())
        for parameter in model.parameters()
    )

def weight_averaging(models):
    model_weights = [get_network_weights(i) for i in models]
    n_sample = len(model_weights)
    weight_averaged = [np.zeros_like(w) for w in model_weights[0]]
    # if sample_interval == 'all':
    for w in model_weights:
        for i in range(len(weight_averaged)):
            weight_averaged[i] += w[i]/n_sample
    return weight_averaged

class BayesCTGANSWA(BaseSynthesizer):
    """Bayesian Conditional Table GAN Synthesizer using Stochastic Weighted Averaging.
    The SWA is only in the Generator, not discriminator (will develop later)

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        loss_discriminator (string):
            The loss function used. Choices are 
            Vanilla loss: 'vanilla' (https://arxiv.org/abs/1406.2661)
            Wasserstein loss: 'wasserstein' (default in CTGAN)
            Least-squares: 'ls' (https://www.arxiv.org/abs/1611.04076)
            f-divergences (https://arxiv.org/abs/1606.00709)
                Total variation: 'f-totalvar'
                forward KLD: 'f-fkl'
                reverse KLD: 'f-rkl'
                Pearson Chi-squared: 'f-chisquare'
                Squared Hellinger: 'f-hellinger'
            Defaults to 'wasserstein' to follow the original CTGAN.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        swa_start (int): 
            SWA start epoch number.
            Defaults to 151.
        swa_lr (float): 
            SWA LR.
            Defaults to 0.002. 
        swa_collect (int):
            SWA model collection frequency/cycle length in epochs.
            Defaults to 1.
        swag (boolean):
            Whether to use SWAG or not. If False, will use SWA instead.
            Defaults to ``False``.
        cov_mat (boolean):
            Whether to save sample covariance during training
            Defaults to ``False``,
        swa_max_models_save (int):
            maximum number of SWAG models to save
            Defaults to 20.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        save_results:
            Whether to save the training results.
            Defaults to ``False``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_opt = Adam, discriminator_opt = Adam,
                 generator_opt_dict = {}, discriminator_opt_dict = {},
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, 
                 batch_size=500, n_test_eval = 5000,
                 country = 'Canada',discriminator_steps=1,
                 log_frequency=True, 
                 loss_discriminator = 'wasserstein',
                 verbose=False, epochs=300, pac=10, 
                 prior_var = 1.0, 
                 swa_start = 151, swa_lr = 2e-3, swa_collect = 1, 
                 cov_mat = False,
                 swag = False,
                 swa_max_models_save = 20,
                 cuda=True,save_results = True):

        assert batch_size % pac == 0, 'batch size should be divisible with Discriminator pac'
        # assert n_samples < epochs, 'number of MCMC samples should be smaller than epochs'
        # assert loss_discriminator in ['vanilla', 'wasserstein','ls'], "loss_discriminator should be between 'vanilla', 'wasserstein' or 'ls'"
        assert swa_start < epochs, "SWA start should be lower than epochs"

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt
        
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self.loss_discriminator = loss_discriminator

        self._batch_size = batch_size
        self.n_test_eval = n_test_eval
        self.country = country

        self._discriminator_steps = discriminator_steps
        
        self.prior_var = prior_var
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_collect = swa_collect
        self.swa_max_models_save = swa_max_models_save
        self.cov_mat = cov_mat
        self.swag = swag
        
        self._log_frequency = log_frequency
        self.save_results = save_results
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def evaluate_test(self, n_data, country):
        real = torch.from_numpy(self._data_sampler.sample_data(n_data, None, None).astype('float32')).to(self._device)
        real = self._apply_activate(real).detach().cpu().numpy()
        real = self._transformer.inverse_transform(real)
        fake = self.sample(n_data,'average')
        roc_val = cal_mean_roc(country,real,fake)
        cio_val = cal_mean_cio(country,real,fake)
        tcap_val = cal_mean_tcap(country,real,fake)
        return [roc_val[0], roc_val[1], cio_val, tcap_val] ## bivariate roc, univariate roc, cio, tcap
    
    def improvement_score(self, n_data, country, best_utility, best_risk):
        eval_values_ = self.evaluate_test(n_data, country)
        overall_utility = sum(eval_values_[:3])/3
        risk = eval_values_[3]
        improv_score = (overall_utility-best_utility)*2 + max(best_risk,0) - max(risk,0)
        return [overall_utility, risk, improv_score]
    
    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        # n_datapoints = torch.tensor(train_data.shape[0])

        generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,bayes=False
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac, loss=self.loss_discriminator,
            bayes=False
        ).to(self._device)

        # print(generator[0].parameters(),log_var_calc_g[0].parameters(),
        #       self._generator_lr,self.generator_opt_dict)

        optimizerG = self.generator_opt(
            generator.parameters(), lr=self._generator_lr,
            betas=(0.5, 0.9), weight_decay=self._generator_decay)

        optimizerD = self.discriminator_opt(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay)
        
        if self.swag:
            self.swag_model = SWAG(Generator(
                                        self._embedding_dim + self._data_sampler.dim_cond_vec(),
                                        self._generator_dim,
                                        data_dim,bayes=False
                                    ).to(self._device),
                                no_cov_mat= not self.cov_mat,
                                max_num_models=self.swa_max_models_save)# if self.swag else deepcopy(self._generator)
        else:
            self.swag_model = Laplace(Generator(
                                        self._embedding_dim + self._data_sampler.dim_cond_vec(),
                                        self._generator_dim,
                                        data_dim,bayes=False
                                    ).to(self._device),no_cov_mat= not self.cov_mat,
                                max_num_models=self.swa_max_models_save)
        # if not self.swag: 
        #     for param in self.swag_model.parameters():
        #         param.detach_()

        gprior_criterion = PriorLoss(prior_std=self.prior_var, 
                                     observed=float(self._batch_size)).to(self._device)
        gnoise_criterion = NoiseLoss(params=generator.parameters(), 
                                     # scale=math.sqrt(2*self._generator_decay/self._generator_lr), 
                                     observed=float(self._batch_size)).to(self._device)

        # print(self.generator_opt,self.discriminator_opt,self._generator_lr,self._discriminator_lr)
        if self.loss_discriminator=='vanilla': 
            loss_func = torch.nn.BCELoss()
        elif self.loss_discriminator=='ls':
            loss_func = torch.nn.MSELoss()

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f}) | Util. ({util:.2f}) | Risk ({risk:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0,util=0, risk=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        self.generator_samples = []
        is_best = [0,0,0] ## initial improvement score, [utility, risk, improvement score]
        for i in epoch_iterator:
            ## prior stdev using gamma distribution (section 5.1)
            # m = torch.distributions.gamma.Gamma(torch.tensor([1.0]), 
            #                                     torch.tensor([1.0]))
            # lambda_gen = 1/m.sample().to(self._device) 
            # lambda_dis = 1/m.sample().to(self._device)
            
            # adjust_learning_rate(optimizerD, self._discriminator_lr, self._epochs, True, self.swa_start, self.swa_lr)
            lr = schedule(i, self._generator_lr, self._epochs, True, self.swa_start, self.swa_lr)
            adjust_learning_rate(optimizerG, lr)
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                
                    fake = generator(fakez)
                    fakeact = self._apply_activate(fake)
                    if c1 is not None:
                        real_cat = torch.cat([real, c2], dim=1)
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    # y_fake = torch.cat([discriminator[mcmc](fake_cat.detach()) for mcmc in self.mcmc_iter[1]])
                    # y_real = torch.cat([discriminator[mcmc](fake_cat.detach()) for mcmc in self.mcmc_iter[1]])
                    y_fake = discriminator(fake_cat.detach())
                    y_real = discriminator(real_cat)
                    # print(y_fake,y_real)
                    real_label = torch.ones_like(y_fake)
                    fake_label = torch.zeros_like(y_real)

                    if self.loss_discriminator=='wasserstein':
                        pen = discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac)
                        loss_d = torch.mean(y_real) - torch.mean(y_fake) + pen
                        # pen.backward(retain_graph=True)
                    elif self.loss_discriminator=='f-totalvar':
                        """ Total Variation """
                        loss_d = -(torch.mean(0.5 * torch.tanh(y_real)) -
                                torch.mean(0.5 * torch.tanh(y_fake)))
                    elif self.loss_discriminator=='f-fkl':
                        """ Forward KL """
                        loss_d = -(torch.mean(y_real) - torch.mean(torch.exp(y_fake - 1)))
                    elif self.loss_discriminator=='f-rkl':
                        """ Reverse KL """
                        loss_d = -(torch.mean(-torch.exp(y_real)) - torch.mean(-1 - y_fake))
                    elif self.loss_discriminator=='f-chisquare':
                        """ Pearson Chi-squared """
                        loss_d = -(torch.mean(y_real) - torch.mean(0.25*y_fake**2 + y_fake))
                    elif self.loss_discriminator=='f-hellinger':
                        """ Squared Hellinger """
                        loss_d = -(torch.mean(1 - torch.exp(y_real)) -
                                torch.mean((1 - torch.exp(y_fake)) / (torch.exp(y_fake))))
                    else:
                        loss_d = loss_func(y_fake,fake_label) + loss_func(y_real, real_label)
                        if self.loss_discriminator=='ls':
                            loss_d /= 2
                        # print(loss_func(y_fake,fake_label), loss_func(y_real, real_label))
                        # print(loss_d)

                    optimizerD.zero_grad(set_to_none=False)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1],dim=1))
                else:
                    y_fake = discriminator(fakeact)
            
                loss_g = torch.tensor(0., device=self._device)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                if self.loss_discriminator=='wasserstein':
                    loss_g += torch.mean(y_fake) + cross_entropy
                elif self.loss_discriminator=='f-totalvar':
                    """ Total Variation """
                    loss_g += -torch.mean(0.5 * torch.tanh(y_fake))
                elif self.loss_discriminator=='f-fkl':
                    """ Forward KL """
                    loss_g += -torch.mean(torch.exp(y_fake - 1))
                elif self.loss_discriminator=='f-rkl':
                    """ Reverse KL """
                    loss_g += -torch.mean(-1 - y_fake)
                elif self.loss_discriminator=='f-chisquare':
                    """ Pearson Chi-squared """
                    loss_g += -torch.mean(0.25*y_fake**2 + y_fake)
                elif self.loss_discriminator=='f-hellinger':
                    """ Squared Hellinger """
                    loss_g += -torch.mean((1 - torch.exp(y_fake)) / (torch.exp(y_fake)))
                else:
                    if self.loss_discriminator=='ls':
                        loss_g += (loss_func(y_fake, real_label)/2) + cross_entropy
                    else:
                        loss_g += loss_func(y_fake, real_label) + cross_entropy

                if self.swag:
                    # print(get_prior(generator.parameters(), self._batch_size))

                    loss_g += gprior_criterion(generator.parameters())
                    loss_g += gnoise_criterion(generator.parameters())

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()
            
            if (i + 1) >= self.swa_start and self.swag:
                # if self.swag: 
                self.swag_model.collect_model(generator)
                # else: update_ema(self.swag_model.parameters(), generator.parameters(), rate = 1/(i+2-self.swa_start))
            
            generator_loss = loss_g.detach().cpu().numpy()
            discriminator_loss = loss_d.detach().cpu().numpy()

            # is_current = self.improvement_score(self.n_test_eval, self.country, is_best[0], is_best[1]) ## [utility, risk, improvement score]
            
            # if self.save_results: 
            #     if is_current[0] > is_best[0]:
            #         is_best = is_current

            #         for mcmc in range(self.mcmc_iter):
            #             torch.save(self._generator[mcmc].state_dict(),
            #                         f'./models_results/BayesGAN_gen_{str(mcmc)}_{self.country}_{self.opt_name}_{self.loss_discriminator}_{i}.pth')
                
            #     if i == epochs-1: ## save final results
            #         for mcmc in range(self.mcmc_iter):
            #             torch.save(self._generator[mcmc].state_dict(),
            #                         f'./models_results/BayesGAN_gen_{str(mcmc)}_{self.country}_{self.opt_name}_{self.loss_discriminator}_{i}.pth')

            is_current = [0,0,0]
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Utility': [is_current[0]],
                'Risk': [is_current[1]],
                'Improvement Score': [is_current[2]]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss,
                                       util=is_current[0], risk=is_current[1],
                                       )
                )

    @random_state
    def sample(self, n, bma_step = 1, scale = 1.0, cov = True, block = False, update_bn = False,
               condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            bma_step (int):
                Number of ensembles for Bayesian model averaging.
                Defaults to 1.
            scale (float):
                Sampling scale.
                Defaults to 1.0.
            cov (boolean):
                Whether to use covariance. Defaults to ``True``.
            block (boolean):
                Whether to use blockwise sampling for covariance. Defaults to False.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        # assert method in ['wa','concat','concat-wa','bma','bma-wa'], "method should be between ['wa','concat','concat-wa','bma','bma']"

        steps = n // self._batch_size + 1
        
        # mean = torch.zeros(n, self._embedding_dim)
        # std = mean + 1
        # fakez = torch.normal(mean=mean, std=std).to(self._device)
        
        data = []
        if scale is None or not self.swag: 
            # print('SWA')
            cov = False
            scale = 0.0
            bma_step = 1 
        self.swag_model.eval()
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fakez_ds = TensorDataset(fakez,fakez) # create your datset
            fakez_loader = DataLoader(fakez_ds, batch_size = self._batch_size) # create your dataloader

            with torch.no_grad():
                for st in range(bma_step):
                    self.swag_model.sample(scale, cov, block = block)
                        # model.eval()
                    if update_bn: bn_update(fakez_loader, self.swag_model)
                    # print('bn updated')
                    fake = self.swag_model(fakez)
                    if st == 0: fakeact = self._apply_activate(fake)
                    else: fakeact += self._apply_activate(fake)
                fakeact /= bma_step
                # print(fakeact)
            
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)