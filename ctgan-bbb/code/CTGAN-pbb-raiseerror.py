"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, Parameter, ReLU, Sequential, functional, Sigmoid

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
# import torchbnn as bnn



class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10, loss='wasserstein', 
                 bayes=False, prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            linearlayer = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=dim, out_features=item) if bayes else Linear(dim, item)
            seq += [linearlayer, LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        if loss=='vanilla': seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o, bayes=False, 
                 prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Residual, self).__init__()
        self.fc = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=i, out_features=o) if bayes else Linear(i, o) 
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, bayes=False, 
                 prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item, bayes, prior_dist, std_act, prior_mu, prior_sigma)]
            dim += item
        linear_layer = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=dim, out_features=data_dim) if bayes else Linear(dim, data_dim)
        seq.append(linear_layer)
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

class CTGAN_bnn(BaseSynthesizer):
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
                 generator_opt = Adam, discriminator_opt = Adam,
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, n_test_eval = 5000, country = 'Canada', discriminator_steps=1,
                 eval_sample = 1,
                 log_frequency=True, loss_discriminator = 'wasserstein',
                 generator_bayes = False, generator_prior_dist='gauss', 
                 generator_std_act='softplus', 
                 generator_prior_mu = 0, generator_prior_sigma = 0.1, 
                 generator_bayes_obj = 'bbb', generator_mc_sample = 1,
                 discriminator_bayes = False, discriminator_prior_dist='gauss', 
                 discriminator_std_act='softplus', 
                 discriminator_prior_mu = 0, discriminator_prior_sigma = 0.1,
                 discriminator_bayes_obj = 'bbb', discriminator_mc_sample = 1,
                 kl_weight = 0.01, delta_pbb = 0.025,
                 verbose=False, epochs=300, pac=10, every = 10,
                 cuda=True,save_results = False, save_folder = ''):
        
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self.generator_opt = generator_opt
        self.opt_name = str(generator_opt).split('.')[-1].replace(r"'>",'')
        self.discriminator_opt = discriminator_opt
        self.loss_discriminator = loss_discriminator
        
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self.n_test_eval = n_test_eval
        self.eval_sample = eval_sample
        self.country = country
        self._discriminator_steps = discriminator_steps
        
        self.generator_bayes = generator_bayes
        self.generator_prior_dist = generator_prior_dist
        self.generator_std_act = generator_std_act
        self.generator_prior_mu = generator_prior_mu
        self.generator_prior_sigma = generator_prior_sigma
        self.generator_bayes_obj = generator_bayes_obj
        self.generator_mc_sample = generator_mc_sample 

        self.discriminator_bayes = discriminator_bayes
        self.discriminator_prior_dist = discriminator_prior_dist
        self.discriminator_std_act = discriminator_std_act
        self.discriminator_prior_mu = discriminator_prior_mu
        self.discriminator_prior_sigma = discriminator_prior_sigma
        self.discriminator_bayes_obj = discriminator_bayes_obj
        self.discriminator_mc_sample = discriminator_mc_sample
        
        self._log_frequency = log_frequency
        self._verbose = verbose
        self.save_results = save_results
        self._epochs = epochs
        self.pac = pac
        self.every = every

        # print(embedding_dim, generator_dim, discriminator_dim,
        #          generator_lr, generator_decay, discriminator_lr,
        #          discriminator_decay, batch_size)

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        # self.kl_weight = torch.Tensor([kl_weight]).to(self._device)
        # self.kl_weight = None if kl_weight is None else torch.tensor(kl_weight, device=self._device)
        self.kl_weight = kl_weight
        self.delta_pbb = torch.tensor(delta_pbb, device=self._device)


        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.folder_name = save_folder
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
        real = self._data_sampler.sample_data(n_data, None, None).astype('float32')
        # real = torch.from_numpy(self._data_sampler.sample_data(n_data, None, None).astype('float32')).to(self._device)
        # real = self._apply_activate(real).detach().cpu().numpy()
        real = self._transformer.inverse_transform(real)
        fake = self.sample(n_data, self.eval_sample)
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
        
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim, self.generator_bayes,
            prior_dist = self.generator_prior_dist, 
            std_act = self.generator_std_act,
            prior_mu= self.generator_prior_mu, 
            prior_sigma= self.generator_prior_sigma 
        ).to(self._device)

        self.discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac, 
            loss=self.loss_discriminator, bayes=self.discriminator_bayes,
            prior_dist = self.discriminator_prior_dist, 
            std_act = self.discriminator_std_act,
            prior_mu= self.discriminator_prior_mu, 
            prior_sigma= self.discriminator_prior_sigma 
        ).to(self._device)

        optimizerG = self.generator_opt(
            self._generator.parameters(), lr=self._generator_lr,
            betas=(0.5, 0.9), weight_decay=self._generator_decay)

        optimizerD = self.discriminator_opt(
            self.discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay)
        
        # for p in discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)
        start = 'B' if self.generator_bayes else ''
        
        if self.loss_discriminator=='vanilla': 
            loss_func = torch.nn.BCELoss()
        elif self.loss_discriminator=='ls':
            loss_func = torch.nn.MSELoss()

        if self.generator_bayes: 
            red_gen = 'mean' if self.generator_bayes_obj == 'bbb' else 'sum'
            loss_kl_gen = BKLLoss(prior_dist = self.generator_prior_dist, 
                                  std_act = self.generator_std_act, 
                                  reduction=red_gen, last_layer_only=False).to(self._device)
            n_data0 = 1.0 if self.generator_bayes_obj == 'bbb' else float(self._batch_size)
            n_data_gen = torch.tensor(n_data0, device=self._device) 
            lambda_gen = Lambda_var(1.0, n_data0).to(self._device) if self.generator_bayes_obj == 'flamb' else None
            lambda_gen_opt = self.generator_opt(
                lambda_gen.parameters(), lr=self._generator_lr,
                betas=(0.5, 0.9), weight_decay=self._generator_decay) if self.generator_bayes_obj == 'flamb' else None
            # loss_kl_gen2 = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(self._device)
            
        if self.discriminator_bayes: 
            red_dis = 'mean' if self.discriminator_bayes_obj == 'bbb' else 'sum'
            loss_kl_dis = BKLLoss(prior_dist = self.discriminator_prior_dist, 
                                  std_act = self.discriminator_std_act, 
                                  reduction=red_dis, last_layer_only=False).to(self._device)
            n_data1 = 1.0 if self.discriminator_bayes_obj == 'bbb' else float(self._batch_size)
            n_data_dis = torch.tensor(n_data1, device=self._device)
            lambda_dis = Lambda_var(1.0, n_data1).to(self._device) if self.discriminator_bayes_obj == 'flamb' else None
            lambda_dis_opt = self.discriminator_opt(
                lambda_dis.parameters(), lr=self._discriminator_lr,
                betas=(0.5, 0.9), weight_decay=self._discriminator_decay) if self.discriminator_bayes_obj == 'flamb' else None
            # loss_kl_dis2 = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(self._device)
            
        
        # print(self.generator_prior_dist, self.generator_std_act, self.generator_bayes_obj,
        #       self.discriminator_prior_dist, self.discriminator_std_act, self.discriminator_bayes_obj)
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss',
                                                 'Utility','Risk','Improvement Score'])

        # print(self.discriminator_mc_sample,self.generator_mc_sample)
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))

        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Disc. ({dis:.2f}) | Util. ({util:.2f}) | Risk ({risk:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0,util=0, risk=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        
        is_best = [0,0,0] ## initial improvement score, [utility, risk, improvement score]
        
        
        # n_data = torch.tensor(len(train_data), device=self._device)
        # print(red_gen,red_dis,n_data_gen,n_data_dis)
        for i in epoch_iterator:
            # prior stdev using gamma distribution (section 5.1)
            # m = torch.distributions.gamma.Gamma(torch.tensor([1.0]), 
            #                                     torch.tensor([1.0]))
            self._generator.train()
            for id_ in range(1,steps_per_epoch+1):
                if self.kl_weight == '2mi':
                    kl_weight = torch.tensor(2**(steps_per_epoch-id_)/(2**(steps_per_epoch)-1), device=self._device)
                elif self.kl_weight == '1m':
                    kl_weight = torch.tensor(1/steps_per_epoch, device=self._device)
                else:
                    kl_weight = self.kl_weight                    
                
                # kl_weight = torch.tensor(1/steps_per_epoch, device=self._device) if self.kl_weight is None else self.kl_weight 
                # torch.tensor(2**(steps_per_epoch-id_+1)/(2**(steps_per_epoch)-1), device=self._device)
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

                    loss_d_total = torch.tensor(0.,device=self._device) 
                    for mc in range(self.discriminator_mc_sample):
                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)
                        
                        if c1 is not None:
                            fake_cat = torch.cat([fakeact, c1], dim=1)
                            real_cat = torch.cat([real, c2], dim=1)
                        else:
                            real_cat = real
                            fake_cat = fakeact

                    ## run monte carlo sample
                    
                        y_fake = self.discriminator(fake_cat)
                        y_real = self.discriminator(real_cat)

                        if torch.isnan(y_fake).sum()>0 or torch.isnan(y_real).sum()>0:
                            raise ValueError('there is NaN when training discriminator')

                        real_label = torch.ones_like(y_fake)
                        fake_label = torch.zeros_like(y_real)
                        ## change from vanilla to ls in 24/9/2024, not in ['bbb','fquad','flamb'] and self.loss_discriminator=='ls'
                        if self.discriminator_bayes and self.discriminator_bayes_obj != 'bbb':
                            y_fake = torch.clamp(y_fake, min=1e-5, max=1.0)
                            y_real = torch.clamp(y_real, min=1e-5, max=1.0)

                        # n_data = torch.tensor(fake.size(0), device=self._device)

                        # print(y_fake, y_real)
                        ## 5 January 2024, changing the loss into negative as in CTGAN real script
                        ## if want to change, remove the negative
                        if self.loss_discriminator=='wasserstein':
                            pen = self.discriminator.calc_gradient_penalty(
                                real_cat, fake_cat, self._device, self.pac)
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake)) + pen
                        
                        ### 8 January 2024, adding f-gan loss
                            
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
                            """Vanilla (vanilla) and least-squares (ls) loss"""
                            loss_d = loss_func(y_fake,fake_label) + loss_func(y_real, real_label)
                            if self.loss_discriminator=='ls':
                                loss_d /= 2

                        # print(loss_d)
                        if self.discriminator_bayes:
                            kl_d = loss_kl_dis(self.discriminator)[0]
                            # kl_d2 = loss_kl_dis2(discriminator)[0]
                            # print(kl_d)
                            
                            loss_d = bayes_total_loss(loss_d, kl_d, n_data_dis, lambda_dis, 
                                                    self.discriminator_bayes_obj, 
                                                    kl_weight, self.delta_pbb, self.discriminator_bayes_obj in ['fquad','flamb'] and self.loss_discriminator == 'vanilla')
                            loss_d = loss_d[0] if self.discriminator_bayes_obj=='flamb' else loss_d
                        
                        loss_d_total += loss_d
                    loss_d_total /= self.discriminator_mc_sample

                    optimizerD.zero_grad(set_to_none=False)
                    if self.discriminator_bayes_obj=='flamb': lambda_dis_opt.zero_grad(set_to_none=False)
                    # if self.loss_discriminator=='wasserstein': pen.backward(retain_graph=True)
                    loss_d_total.backward()
                    optimizerD.step()
                    if self.discriminator_bayes_obj=='flamb': lambda_dis_opt.step()
                    

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                loss_g_total = torch.tensor(0.,device=self._device) 
                for mc in range(self.generator_mc_sample):
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    # n_data = torch.tensor(fake.size(0), device=self._device)
                    if c1 is not None:
                        y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = self.discriminator(fakeact)

                    if torch.isnan(y_fake).sum()>0:
                        raise ValueError('there is NaN when training generator')

                    if self.generator_bayes and self.generator_bayes_obj != 'bbb':
                        y_fake = torch.clamp(y_fake, min=1e-5, max=1.0)

                    if condvec is None:
                        cross_entropy = 0
                    else:
                        cross_entropy = self._cond_loss(fake, c1, m1)

                    if self.loss_discriminator=='wasserstein':
                        loss_g = -torch.mean(y_fake)
                        
                    ### 8 January 2024, adding f-gan loss 
                    
                    elif self.loss_discriminator=='f-totalvar':
                        """ Total Variation """
                        loss_g = -torch.mean(0.5 * torch.tanh(y_fake))
                    elif self.loss_discriminator=='f-fkl':
                        """ Forward KL """
                        loss_g = -torch.mean(torch.exp(y_fake - 1))
                    elif self.loss_discriminator=='f-rkl':
                        """ Reverse KL """
                        loss_g = -torch.mean(-1 - y_fake)
                    elif self.loss_discriminator=='f-chisquare':
                        """ Pearson Chi-squared """
                        loss_g = -torch.mean(0.25*y_fake**2 + y_fake)
                    elif self.loss_discriminator=='f-hellinger':
                        """ Squared Hellinger """
                        loss_g = -torch.mean((1 - torch.exp(y_fake)) / (torch.exp(y_fake)))
                    else:
                        """Vanilla and least-squares loss"""
                        loss_g = loss_func(y_fake, real_label)
                        if self.loss_discriminator=='ls':
                            loss_g /= 2
                        
                    loss_g += cross_entropy
                    
                    # loss_g = -torch.mean(y_fake) + cross_entropy

                    if self.generator_bayes:
                        kl_g = loss_kl_gen(self._generator)[0]
                        # kl_g2 = loss_kl_gen2(self._generator)[0]
                        # print(kl_g)

                        # raise RuntimeError('stop first')
                        loss_g = bayes_total_loss(loss_g, kl_g, n_data_gen, lambda_gen, 
                                                self.generator_bayes_obj, 
                                                kl_weight, self.delta_pbb, 
                                                self.generator_bayes_obj in ['fquad','flamb'] and self.loss_discriminator == 'vanilla')
                        
                        loss_g = loss_g[0] if self.generator_bayes_obj=='flamb' else loss_g
                    loss_g_total += loss_g
                loss_g_total /= self.generator_mc_sample
                # print(loss_d,kl_d,n_data_dis,loss_d_total,loss_g,kl_g,n_data_gen,loss_g_total)

                optimizerG.zero_grad(set_to_none=False)
                if self.generator_bayes_obj=='flamb': lambda_gen_opt.zero_grad(set_to_none=False)
                loss_g_total.backward()
                optimizerG.step()
                if self.generator_bayes_obj=='flamb': lambda_gen_opt.step()
            
            generator_loss = loss_g_total.detach().cpu().numpy()
            discriminator_loss = loss_d_total.detach().cpu().numpy()

            is_current = self.improvement_score(self.n_test_eval, self.country, is_best[0], is_best[1]) if self.n_test_eval is not None else [0,0,0] ## [utility, risk, improvement score]
            # is_current = self.improvement_score(self.n_test_eval, self.country, is_best[0], is_best[1]) ## [utility, risk, improvement score]
            if ((i+1)%self.every==0 or i == epochs-1) and self.n_test_eval is None:
                is_current = self.improvement_score(len(train_data), self.country, is_best[0], is_best[1]) ## [utility, risk, improvement score]
            

#             if self.save_results: 
#                 if is_current[0] > is_best[0] and self.n_test_eval is not None:
#                     is_best = is_current
#                     torch.save(self._generator.state_dict(),
#                                 f'{self.folder_name}/model_best.pth')
                
#                 if i == epochs-1: ## save final results
#                     torch.save(self._generator.state_dict(),
#                                 f'{self.folder_name}/model_last.pth')

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
                                        util=is_current[0], risk=is_current[1])
                )
            
            if self.save_results: 
                obj1 = self.generator_bayes_obj if self.generator_bayes else 'ctgan'
                loss1 = self.loss_discriminator
                # ep1 = str(epochs)
                # s1 = self.generator_prior_sigma
                # dis1 = str(self.discriminator_bayes)
                # mc1 = str(self.generator_mc_sample)

                self.loss_values.to_csv(f'{self.folder_name}/loss_values_{self.country}_{obj1}_{loss1}.csv', index=False)


    @random_state
    def sample(self, n, mc_sample = 1, update_bn = False, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
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

        steps = n // self._batch_size + 1
        data = []
        self._generator.eval()
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

            # fake = self._generator(fakez)
            # fakeact = self._apply_activate(fake)
            with torch.no_grad():
                for mc in range(mc_sample):
                    if update_bn: 
                        freeze_weight(self._generator)
                        bn_update(fakez_loader, self._generator)
                        fake = self._generator(fakez)
                        unfreeze_weight(self._generator)
                    else:
                        fake = self._generator(fakez)
                    if mc == 0:
                        fakeact = self._apply_activate(fake)
                    else:
                        fakeact += self._apply_activate(fake)
                fakeact /= mc_sample
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)