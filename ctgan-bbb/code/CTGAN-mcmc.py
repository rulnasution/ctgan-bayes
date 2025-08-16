"""CTGAN module."""

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

from optimiser_mcmc import *

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

class BayesCTGANSaatchi(BaseSynthesizer):
    """Bayesian Conditional Table GAN Synthesizer.

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
        generator_opt_dict / discriminator_opt_dict:
            Dictionary for the optimizer in generator and discriminator.
            Choices are between 'Adam', 'SGLD','PSGLD', 'SGHMC', and 'ASGHMC'
            source for MCMC optimizer is https://github.com/automl/pybnn/tree/master/pybnn/sampler
            and for prior loss functions are https://github.com/vasiloglou/mltrain-nips-2017/blob/master/ben_athiwaratkun/pytorch-bayesgan/Bayesian%20GAN%20in%20PyTorch.ipynb 
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
        mcmc_iter (int):
            Number of MCMC chains.
            Defaults to 1
        n_samples (int):
            Number of MCMC samples to collect. Defaults to 10.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_opt_dict = {}, discriminator_opt_dict = {},
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, 
                 batch_size=500, n_test_eval = 5000, full_eval_every = 25,
                 country = 'Canada',discriminator_steps=1,
                 log_frequency=True, 
                 loss_discriminator = 'wasserstein',
                 mcmc_iter = 1, ## The number of MCMC chains to run in parallel
                 n_samples = 10, ## no.samples to take for data generation, should be less than epoch
                 init_lr = 2e-3,
                 verbose=False, epochs=300, pac=10, cuda=True,save_results = False,
                 save_folder = ''):

        assert batch_size % 2 == 0 and batch_size % pac == 0, 'batch size should be divisible by 2 and pac'
        assert n_samples < epochs, 'number of MCMC samples should be smaller than epochs'
        # assert loss_discriminator in ['vanilla', 'wasserstein','ls'], "loss_discriminator should be between 'vanilla', 'wasserstein' or 'ls'"

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self.generator_opt_dict = generator_opt_dict
        self.discriminator_opt_dict = discriminator_opt_dict
        
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self.init_lr = init_lr
        self.loss_discriminator = loss_discriminator

        self._batch_size = batch_size
        self.n_test_eval = n_test_eval
        self.country = country
        self.full_eval_every = full_eval_every

        self._discriminator_steps = discriminator_steps
        self.mcmc_iter = [mcmc_iter] if isinstance(mcmc_iter, int) else mcmc_iter
        self.mcmc_iter = [self.mcmc_iter[0],self.mcmc_iter[0]] if len(self.mcmc_iter)==1 else self.mcmc_iter
        self.n_samples = n_samples
        # self.generator_drag_term = generator_drag_term
        # self.discriminator_drag_term = discriminator_drag_term

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
        self.save_folder = save_folder

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

    def evaluate_test_compre(self, n_data, country):
        # real = torch.from_numpy(self._data_sampler.sample_data(n_data, None, None).astype('float32')).to(self._device)
        # real = self._apply_activate(real).detach().cpu().numpy()
        # real2 = self._transformer.inverse_transform(real)
        if n_data != len(self.train_data): real2 = self.train_data.sample(n_data).reset_index(drop=True)

        results = []
        # print(self.eval_swa)

        for method in ['wa','concat','concat-wa','bma','bma-wa']: ## MCMC
            # print(real.head())
            # print(real.info())
            real = deepcopy(self.train_data) if n_data == len(self.train_data) else deepcopy(real2)
            fake = self.sample(n_data, method)
            eval_res = self.evaluate_test(real, fake, country)
            results.append([self.generator_opt_dict['optimiser'], method, len(self.generator_samples)] + eval_res)
        
        results1 = pd.DataFrame(results)
        results1.columns = ['method','sampling_method','mcmc_samples',
                            'roc_uni','roc_bi', 'cio','utility','risk']
        return results1

    def evaluate_test(self, real, fake, country):
        roc_val = cal_mean_roc(country,real,fake)
        cio_val = cal_mean_cio(country,real,fake)
        tcap_val = cal_mean_tcap(country,real,fake)
        return [roc_val[0], roc_val[1], cio_val, (roc_val[0]+roc_val[1]+cio_val)/3, tcap_val] ## bivariate roc, univariate roc, cio, tcap
    
    # def evaluate_test(self, n_data, country):
    #     real = torch.from_numpy(self._data_sampler.sample_data(n_data, None, None).astype('float32')).to(self._device)
    #     real = self._apply_activate(real).detach().cpu().numpy()
    #     real = self._transformer.inverse_transform(real)
    #     fake = self.sample(n_data,'average')
    #     roc_val = cal_mean_roc(country,real,fake)
    #     cio_val = cal_mean_cio(country,real,fake)
    #     tcap_val = cal_mean_tcap(country,real,fake)
    #     return [roc_val[0], roc_val[1], cio_val, tcap_val] ## bivariate roc, univariate roc, cio, tcap
    
    # def improvement_score(self, n_data, country, best_utility, best_risk):
    #     eval_values_ = self.evaluate_test(n_data, country)
    #     overall_utility = sum(eval_values_[:3])/3
    #     risk = eval_values_[3]
    #     improv_score = (overall_utility-best_utility)*2 + max(best_risk,0) - max(risk,0)
    #     return [overall_utility, risk, improv_score]
    
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
        self.train_data = train_data
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
        
        generator = [Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,bayes=False
        ).to(self._device) for i in range(self.mcmc_iter[0])]

        discriminator = [Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac, loss=self.loss_discriminator,
            bayes=False
        ).to(self._device) for i in range(self.mcmc_iter[1])]

        self._generator_samples = []

        # log_var_calc_g = [log_variance(noise = self.generator_opt_dict['var_noise']).to(self._device) for i in range(self.mcmc_iter[0])]
        # log_var_calc_d = [log_variance(noise = self.discriminator_opt_dict['var_noise']).to(self._device) for i in range(self.mcmc_iter[1])]
        
        n_datapoints = len(self.train_data)
        gprior_criterion = PriorLoss(prior_std=math.sqrt(self.generator_opt_dict['prior_var']), 
                                     observed=float(n_datapoints)).to(self._device) 
        # gnoise_criterion = NoiseLoss(params=generator[0].parameters(), 
        #                              scale=math.sqrt(2*self._generator_decay/self._generator_lr), 
        #                              observed=float(self._batch_size)).to(self._device)
        
        dprior_criterion = PriorLoss(prior_std=math.sqrt(self.discriminator_opt_dict['prior_var']), 
                                     observed=float(n_datapoints)).to(self._device) if self.discriminator_opt_dict['optimiser'] != 'Adam' else None
        # dnoise_criterion = NoiseLoss(params=discriminator[0].parameters(), 
        #                              scale=math.sqrt(2*self._discriminator_decay/self._discriminator_lr), 
        #                              observed=float(self._batch_size)).to(self._device)

        # print(generator[0].parameters(),log_var_calc_g[0].parameters(),
        #       self._generator_lr,self.generator_opt_dict)

        optimizerG = [get_mcmc_optimiser(
            list(generator[i].parameters()),# + list(log_var_calc_g[i].parameters()), 
            lr=self._generator_lr,
            params_dict=self.generator_opt_dict) for i in range(self.mcmc_iter[0])]

        optimizerD = [get_mcmc_optimiser(
            list(discriminator[i].parameters()),# + list(log_var_calc_d[i].parameters()), 
            lr=self._generator_lr,
            params_dict=self.discriminator_opt_dict) for i in range(self.mcmc_iter[1])]
        
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
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})' #  | Util. ({util:.2f}) | Risk ({risk:.2f})
            epoch_iterator.set_description(description.format(gen=0, dis=0)) # util=0, risk=0

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        self.generator_samples = []
        # is_best = [0,0,0] ## initial improvement score, [utility, risk, improvement score]
        if 'SGLD' in self.generator_opt_dict['optimiser']:
            a,b = finding_a_b_maxteh(self.init_lr, self._generator_lr, epochs, 0.55)
        
        self.epsilons = []
        self.evaluation_results = pd.DataFrame()
        for i in epoch_iterator:
            ## prior stdev using gamma distribution (section 5.1)
            # m = torch.distributions.gamma.Gamma(torch.tensor([1.0]), 
            #                                     torch.tensor([1.0]))
            # lambda_gen = 1/m.sample().to(self._device) 
            # lambda_dis = 1/m.sample().to(self._device)
            if 'SGLD' in self.generator_opt_dict['optimiser']:
                lr = schedule_maxteh(a, b, i, 0.55)
                self.epsilons.append(lr)
                for mcmc in range(self.mcmc_iter[0]): adjust_learning_rate(optimizerG[mcmc], lr) 
            if 'SGLD' in self.discriminator_opt_dict['optimiser']:
                for mcmc in range(self.mcmc_iter[1]): adjust_learning_rate(optimizerD[mcmc], lr) 
            
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    fakezv = torch.autograd.Variable(fakez)

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
                    fakes = []
                    reals = []

                    for mcmc in range(self.mcmc_iter[0]):
                        generator[mcmc].train()
                        fake = generator[mcmc](fakez)
                        fakeact = self._apply_activate(fake)
                        if c1 is not None:
                            reals.append(torch.cat([real, c2], dim=1))
                            fakes.append(torch.cat([fakeact, c1], dim=1))
                        else:
                            reals.append(real)
                            fakes.append(fakeact)

                    fake_cat = torch.cat(fakes)                        
                    real_cat = torch.cat(reals)                        

                    loss_d = torch.tensor(0., device=self._device)
                    # y_fake = torch.cat([discriminator[mcmc](fake_cat.detach()) for mcmc in self.mcmc_iter[1]])
                    # y_real = torch.cat([discriminator[mcmc](fake_cat.detach()) for mcmc in self.mcmc_iter[1]])
                    for mcmc in range(self.mcmc_iter[1]):
                        y_fake = discriminator[mcmc](fake_cat.detach())
                        y_real = discriminator[mcmc](real_cat)
                        # print(y_fake,y_real)
                        real_label = torch.ones_like(y_fake)
                        fake_label = torch.zeros_like(y_real)
                        
                        if self.loss_discriminator=='wasserstein':
                            pen = discriminator[mcmc].calc_gradient_penalty(
                                real_cat, fake_cat, self._device, self.pac)
                            loss_d1 = -(torch.mean(y_real) - torch.mean(y_fake)) + pen
                            # pen.backward(retain_graph=True)
                        elif self.loss_discriminator=='f-totalvar':
                            """ Total Variation """
                            loss_d1 = -(torch.mean(0.5 * torch.tanh(y_real)) -
                                    torch.mean(0.5 * torch.tanh(y_fake)))
                        elif self.loss_discriminator=='f-fkl':
                            """ Forward KL """
                            loss_d1 = -(torch.mean(y_real) - torch.mean(torch.exp(y_fake - 1)))
                        elif self.loss_discriminator=='f-rkl':
                            """ Reverse KL """
                            loss_d1 = -(torch.mean(-torch.exp(y_real)) - torch.mean(-1 - y_fake))
                        elif self.loss_discriminator=='f-chisquare':
                            """ Pearson Chi-squared """
                            loss_d1 = -(torch.mean(y_real) - torch.mean(0.25*y_fake**2 + y_fake))
                        elif self.loss_discriminator=='f-hellinger':
                            """ Squared Hellinger """
                            loss_d1 = -(torch.mean(1 - torch.exp(y_real)) -
                                    torch.mean((1 - torch.exp(y_fake)) / (torch.exp(y_fake))))
                        else:
                            loss_d1 = loss_func(y_fake,fake_label) + loss_func(y_real, real_label)
                            if self.loss_discriminator=='ls':
                                loss_d1 /= 2
                            # print(loss_func(y_fake,fake_label), loss_func(y_real, real_label))
                            # print(loss_d)
                        
                        if self.discriminator_opt_dict['optimiser'] != 'Adam':
                            # log_variance1 = log_var_calc_d[mcmc](fake)
                            # log_prediction_variance = log_variance1.view((-1, 1))
                            # prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)
                            # loss_var = log_variance_prior(log_variance1, mean = self.discriminator_opt_dict['prior_mean'] + 1e-6, 
                            #                             variance = self.discriminator_opt_dict['prior_var'])
                            # loss_prior = weight_prior(list(discriminator[mcmc].parameters()) + list(log_var_calc_d[mcmc].parameters()), 
                            #                         wdecay = self._generator_decay, device = self._device)
                            # loss_bayes = (loss_var + loss_prior) / n_datapoints
                            # loss_bayes = get_prior(discriminator[mcmc].parameters(), self._batch_size)
                            loss_bayes = dprior_criterion(discriminator[mcmc].parameters()) #  + dnoise_criterion(discriminator[mcmc].parameters()

                            loss_d += loss_d1 + loss_bayes
                        else:
                            loss_d += loss_d1 

                    loss_d /= self.mcmc_iter[1]
                    for mcmc in range(self.mcmc_iter[1]): optimizerD[mcmc].zero_grad(set_to_none=False)
                    loss_d.backward()
                    for mcmc in range(self.mcmc_iter[1]): optimizerD[mcmc].step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fakes = []
                fakeacts = []
                y_fakes = []
                loss_g = 0
                for mcmc in range(self.mcmc_iter[0]):
                    fake = generator[mcmc](fakez)
                    fakes.append(fake)
                    fakeact = self._apply_activate(fake)
                    fakeacts.append(fakeact)
                fake = torch.cat(fakes)
                fakeact = torch.cat(fakeacts)

                for mcmc in range(self.mcmc_iter[1]):
                    if c1 is not None:
                        y_fakes.append(discriminator[mcmc](torch.cat([fakeact, torch.cat([c1]*self.mcmc_iter[0])], dim=1)))
                    else:
                        y_fakes.append(discriminator[mcmc](fakeact))
                
                y_fake = torch.cat(y_fakes)
                
                loss_g = torch.tensor(0., device=self._device)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(torch.cat([fake]*self.mcmc_iter[1]), torch.cat([c1]*self.mcmc_iter[0]*self.mcmc_iter[1]), 
                                                    torch.cat([m1]*self.mcmc_iter[0]*self.mcmc_iter[1]))

                if self.loss_discriminator=='wasserstein':
                    loss_g += -torch.mean(y_fake) + cross_entropy
                elif self.loss_discriminator=='f-totalvar':
                    """ Total Variation """
                    loss_g = -torch.mean(0.5 * torch.tanh(y_fake)) + cross_entropy
                elif self.loss_discriminator=='f-fkl':
                    """ Forward KL """
                    loss_g = -torch.mean(torch.exp(y_fake - 1)) + cross_entropy
                elif self.loss_discriminator=='f-rkl':
                    """ Reverse KL """
                    loss_g = -torch.mean(-1 - y_fake) + cross_entropy
                elif self.loss_discriminator=='f-chisquare':
                    """ Pearson Chi-squared """
                    loss_g = -torch.mean(0.25*y_fake**2 + y_fake) + cross_entropy
                elif self.loss_discriminator=='f-hellinger':
                    """ Squared Hellinger """
                    loss_g = -torch.mean((1 - torch.exp(y_fake)) / (torch.exp(y_fake))) + cross_entropy
                else:
                    if self.loss_discriminator=='ls':
                        loss_g += (loss_func(y_fake, real_label)/2) + cross_entropy
                    else:
                        loss_g += loss_func(y_fake, real_label) + cross_entropy
                
                for mcmc in range(self.mcmc_iter[0]):
                    # log_variance1 = log_var_calc_g[mcmc](fake)
                    # loss_var = log_variance_prior(log_variance1, mean = self.generator_opt_dict['prior_mean'] + 1e-6, 
                    #                             variance = self.generator_opt_dict['prior_var'])
                    # loss_prior = weight_prior(list(generator[mcmc].parameters()) + list(log_var_calc_g[mcmc].parameters()), 
                    #                         wdecay = self._generator_decay, device = self._device)
                    # loss_bayes = (loss_var + loss_prior) / n_datapoints
                    # loss_bayes = get_prior(self._generator[mcmc].parameters(), self._batch_size) 
                    # loss_g += loss_bayes
                    loss_g += gprior_criterion(generator[mcmc].parameters())
                    # loss_g += gnoise_criterion(generator[mcmc].parameters())

                loss_g /= self.mcmc_iter[0]

                for mcmc in range(self.mcmc_iter[0]): optimizerG[mcmc].zero_grad(set_to_none=False)
                loss_g.backward()
                for mcmc in range(self.mcmc_iter[0]): optimizerG[mcmc].step()
            
            self._generator_samples.append(generator)
            if len(self._generator_samples) > self.n_samples:
                self._generator_samples.pop(0) ## remove the previous sample
                if 'SGLD' in self.generator_opt_dict['optimiser']: self.epsilons.pop(0) ## remove the previous sample
            
            self.generator_averaged_all = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,bayes=False
            ).to(self._device)

            g_weight_averaged_all = weight_averaging(list(itertools.chain(*self._generator_samples)))
            with torch.no_grad():
                for parameter, sample in zip(self.generator_averaged_all.parameters(), g_weight_averaged_all):
                    parameter.copy_(torch.from_numpy(sample))
            
            self.generator_averaged_by_chain = [Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,bayes=False
            ).to(self._device) for mcmc in range(self.mcmc_iter[0])]

            self.discriminator = discriminator

            for mcmc in range(self.mcmc_iter[0]):
                generator_1 = [sample[mcmc] for sample in self._generator_samples]
                g_weight_averaged = weight_averaging(generator_1)
                with torch.no_grad():
                    for parameter, sample in zip(self.generator_averaged_by_chain[mcmc].parameters(), 
                                                g_weight_averaged):
                        parameter.copy_(torch.from_numpy(sample))

            
            # if (self._epochs - i) <= self.n_samples:
            #     self._generator_samples.append(generator)

            generator_loss = loss_g.detach().cpu().numpy()
            discriminator_loss = loss_d.detach().cpu().numpy()

            if self.save_results:
                if ((i+1)%self.full_eval_every ==0 or i in [0, epochs-1]):
                    eval_current = self.evaluate_test_compre(len(train_data), self.country) ## [utility, risk, improvement score]
                else:
                    eval_current = self.evaluate_test_compre(self.n_test_eval, self.country) if self.n_test_eval is not None else None
                # print(eval_current)
                if eval_current is not None:
                    eval_current['epoch'] = i
                    self.evaluation_results = pd.concat([self.evaluation_results, eval_current], axis=0)
                    gen1 = self.generator_opt_dict['optimiser'] + '_' + str(self.mcmc_iter[0])
                    dis1 = self.discriminator_opt_dict['optimiser'] + '_' + str(self.mcmc_iter[1])
                    self.evaluation_results.to_csv(f'{self.save_folder}/MCMC_{gen1}_{dis1}_eval_results_{self.country}_{self.loss_discriminator}_{str(self._epochs)}.csv', index=False)

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

            # is_current = [0,0,0]
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
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
                                       # util=is_current[0], risk=is_current[1],
                                       )
                )

    @random_state
    def sample(self, n, method='concat', condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            method (string):
                Method used to sample data. Combination between 'concat' or 'bma' and 'wa' or 'ma'. 
                    'wa'=weight averaging, 'ma'=model averaging
                    wa is by averaging all posterior samples regardless of chains
                    concat is by concatenating data using the samples from posterior regardless of chains
                    concat-wa is by averaging based on the MCMC chains then do concat
                    bma is by doing Bayesian model averaging using the samples from posterior
                    bma-wa is by averaging based on the MCMC chains then do bma
                Defaults to 'concat'.
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
        '''
        wa=weight averaging, ma=model averaging
        wa is by averaging all posterior samples regardless of chains
        concat is by concatenating data using the samples from posterior regardless of chains
        concat-wa is by averaging based on the MCMC chains then do concat
        bma is by doing Bayesian model averaging using the samples from posterior
        bma-wa is by averaging based on the MCMC chains then do bma
        '''

        assert method in ['wa','concat','concat-wa','bma','bma-wa'], "method should be between ['wa','concat','concat-wa','bma','bma']"

        steps = n // self._batch_size + 1
        if method in ['concat','bma']: _generator = list(itertools.chain(*self._generator_samples))
        elif method=='wa': _generator = self.generator_averaged_all
        elif '-wa' in method: _generator = self.generator_averaged_by_chain

        if 'concat' in method: 
            idx = np.random.randint(0,len(_generator),steps)
        if method != 'wa': m = len(_generator)
        data = []
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
            if 'concat' in method:
                with torch.no_grad(): fake = _generator[idx[i]](fakez)
            elif 'bma' in method:
                with torch.no_grad():
                    for idx in range(m):
                        fake = _generator[idx](fakez)
                        if idx == 0: fakeact = self._apply_activate(fake)
                        else: 
                            eps = self.epsilons[idx] if len(self.epsilons) > 0 else 1.0
                            fakeact += eps*self._apply_activate(fake)
                    if len(self.epsilons) == 0: fakeact /= self.mcmc_iter[0]
                    else: fakeact /= sum(self.epsilons)
            elif method=='wa':
                with torch.no_grad(): fake = _generator(fakez)
            fakeact = self._apply_activate(fake) if 'bma' not in method else fakeact
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)