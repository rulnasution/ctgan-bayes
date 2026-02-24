import torch, numpy as np, copy, math
from torch.optim import Optimizer, SGD, Adam
from typing import Iterable
from torch import optim

def get_mcmc_optimiser(parameters, lr, params_dict):
    if params_dict['optimiser'] == 'Adam':
        return Adam(parameters, lr=lr,**{i:params_dict[i] for i in ['betas','weight_decay']})
    elif params_dict['optimiser'] == 'SGLD':
        return SGLD(parameters, lr=lr,**{i:params_dict[i] for i in ['scale_grad']})
    elif params_dict['optimiser'] == 'PSGLD':
        return PreconditionedSGLD(parameters, lr=lr,**{i:params_dict[i] for i in ['num_train_points','precondition_decay_rate','diagonal_bias']})
    elif params_dict['optimiser'] == 'SGHMC':
        return SGHMC(parameters, lr=lr,**{i:params_dict[i] for i in ['mdecay','wd','scale_grad']})
    elif params_dict['optimiser'] == 'ASGHMC':
        return AdaptiveSGHMC(parameters, lr=lr,**{i:params_dict[i] for i in ['num_burn_in_steps', 'epsilon', 'mdecay', 'scale_grad']})
    elif params_dict['optimiser'] == 'KFAC':
        return KFACLaplace(parameters, **{i:params_dict[i] for i in ['eps', 'sua', 'pi', 'update_freq', 
                                                                     'alpha', 'constraint_norm','data_size',
                                                                     'use_batch_norm']})
    elif params_dict['optimiser'] == 'CoinBP':
        return CocobBackprop(parameters)
    elif params_dict['optimiser'] == 'CoinONS':
        return CocobOns(parameters)
    else:
        raise ValueError('No optimiser named "'+ params_dict['optimiser'] + '"')

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def schedule_maxteh(a,b,t,dr = 0.55):
    '''
    Polynomial decay learning rate schedule for SGLD.
    Source: 
    [1] https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf Welling and Teh 2011 SGLD, the text is after equation 2.
    a and b is constant
    t is the iteration
    dt is the decay rate (gamma) about (0.5, 1]
    '''
    second = (b+t)**(-dr)
    return a/second

def finding_a_b_maxteh(init_lr, final_lr, epochs, dr = 0.55):
    '''
    finding a and b from given learning rate in SGLD paper
    '''
    first_term = epochs * (init_lr**dr)
    second_term = 1-(init_lr**dr)
    b = first_term/second_term
    a = final_lr * ((b+epochs)**dr)
    return a,b

def schedule_cyclical_sgmcmc(init_lr, t, m_cycles, epochs):
    '''
    Cosine learning rate schedule for cyclical SGMCMC.
    Source: 
    [1] https://arxiv.org/pdf/1902.03932 Zhang et al. 2020 Cyclical SGMCMC, equation 1.
    needs initial learning rate (init_lr), current iteration (t),
    number of cycles (m_cycles), epochs
    '''
    first_term = init_lr/2
    km = np.ceil(epochs/m_cycles)
    second_term = (t % km) / km
    return first_term * (np.cos(np.pi * second_term) + 1)

class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler
    """

    def __init__(self,
                 params,
                 lr: np.float64 = 1e-2,
                 scale_grad: np.float64 = 1) -> None:

        """ Set up a SGLD Optimizer. 
        source: 
        [1] https://doi.org/10.1080/01621459.2020.1847120 Nemeth and Fearnhead 2021 SGMCMC
        [2] https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf Welling and Teh 2011 SGLD

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            
        defaults = dict(
            lr=lr,
            scale_grad=scale_grad
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr, scale_grad = group["lr"], group["scale_grad"]
                # the average gradient over the batch, i.e N/n sum_i g_theta_i + g_prior
                ## parameter.grad.data = delta U(theta) scale grad = N/n (according to equation 4)
                ## but, if the loss has been averaged already, only need to define the N 
                gradient = parameter.grad.data * scale_grad
                #  State initialization
                if len(state) == 0:
                    state["iteration"] = 0
                ## sigma = sqrt(h)
                sigma = torch.sqrt(torch.from_numpy(np.array(lr, dtype=type(lr))))
                ## right hand component of theta updating in algorithm 1
                ## basic equation is in equation 2, can convert dtheta into theta(k+1)-theta(k)
                ## 0.5*lr = h_k/2, gradient = delta U(theta_k), xi_k = sigma*normal(0,1)
                delta = (0.5 * lr * gradient +
                         sigma * torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)))
                ## update theta k+1
                parameter.data.add_(-delta)
                state["iteration"] += 1
                state["sigma"] = sigma

        return loss

class PreconditionedSGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in each dimension
        according to RMSProp.
        Source: 
        [1] Li et al. 2015 https://arxiv.org/pdf/1512.07666.pdf
    """
    def __init__(self,
                 params,
                 lr=np.float64(1e-2),
                 num_train_points=1,
                 precondition_decay_rate=np.float64(0.99),
                 diagonal_bias=np.float64(1e-5)) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.99`
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-5`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            
        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            diagonal_bias=diagonal_bias,
            num_train_points=num_train_points
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_train_points = group["num_train_points"] ## n
                precondition_decay_rate = group["precondition_decay_rate"]  # alpha
                diagonal_bias = group["diagonal_bias"]  # lambda
                ## kind a different, the n is not inversed
                ## while in the algorithm it's divided by n
                ## parameter.grad.data = delta p(d_t|theta_t) the sample mean of the gradient using mini-batch
                gradient = parameter.grad.data * num_train_points 

                #  state initialization
                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                state["iteration"] += 1

                #  momentum update V(theta_t)
                ## according to algorithm 1
                ## momentum = V(theta t-1) precondition_decay_rate = alpha, gradient**2 = g(theta;D) \odot g(theta;D) 
                momentum = state["momentum"]
                momentum_t = momentum * precondition_decay_rate + (1.0 - precondition_decay_rate) * (gradient ** 2)
                state["momentum"] = momentum_t  # V(theta_t+1)

                # compute preconditioner G(theta_t)
                # diagonal bias = lambda 1, 1 is vector ones -> in code we do not need to put the ones
                preconditioner = (1. / (torch.sqrt(momentum_t) + diagonal_bias))  # G(theta_t+1)

                # standard deviation of the injected noise
                ## lr = epsilon_t
                ## last term of equation (3), the epsilon_t is square rooted with G(theta)
                ## sigma = sqrt(epsilon_t * G(theta))
                ## since torch requires std, and the paper stated variance in the normal distribution, we just sqrt it
                sigma = torch.sqrt(torch.from_numpy(np.array(lr, dtype=type(lr)))) * torch.sqrt(preconditioner)
                ## in the paper, the gradient term is (gradient of log p(theta))+ N* g(theta;D)
                ## but the code did not ptovide it. is the gradient include N g(theta)
                ## there also should be gamma_i(theta) which is the summation across j 
                ## of gradient G(theta) w/ respect to theta_j (ased on equation 3)
                mean = 0.5 * lr * (preconditioner * gradient)
                delta = (mean + torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)) * sigma)

                parameter.data.add_(-delta)

        return loss
    
class SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.
        See [1] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.
        [1] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    def __init__(self,
                 params,
                 lr: float=1e-2,
                 mdecay: float=0.01,
                 wd: float=0.00002,
                 scale_grad: float=1.) -> None:
        """ Set up a SGHMC Optimizer.
        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset. Equal to n_datapoints
            Default: `1.0`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            
        defaults = dict(
            lr=lr, scale_grad=scale_grad,
            mdecay=mdecay,
            wd=wd
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                ## sample r -> momentum, since use randn, 
                ## M is assumed to be 1 since it needs to be estimated via riemann geometry
                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.randn(parameter.size(), dtype=parameter.dtype).to(self.device)

                state["iteration"] += 1
                ## directly simulate dynamics based on equation 13
                ## obtain momentum decay, lr, and weight decay
                ## lr = epsilon_t, mdecay = C 
                ## (C-B)*epsilon_t = diffusion matrix B(theta), according to equation 7 in dr
                mdecay, lr, wd = group["mdecay"], group["lr"], group["wd"]
                scale_grad = group["scale_grad"]

                momentum = state["momentum"]
                ## delta U(theta_i)
                gradient = parameter.grad.data * scale_grad
                
                ### sigma = sqrt(2(C-B)epsilon_t), but there is no B in here
                ### the normal distribution parameter is the mean and variance, therefore need to sqrt
                sigma = torch.sqrt(torch.from_numpy(np.array(2 * lr * mdecay, dtype=type(lr))))
                sample_t = torch.normal(mean=torch.zeros_like(gradient),std=torch.ones_like(gradient)*sigma).to(self.device)
                theta_update = torch.tensor(lr * mdecay).to(self.device)
                parameter.data.add_(theta_update*momentum) ## \theta_i
                r_update = -lr * gradient - theta_update*momentum + sample_t
                momentum.add_(r_update) ## r_i
        return loss
    
class AdaptiveSGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. 
            <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 num_burn_in_steps: int = 3000,
                 epsilon: float = 1e-16,
                 mdecay: float = 0.05,
                 scale_grad: float = 1.) -> None:
        """ Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        epsilon: float, optional
            (Constant) per-parameter epsilon level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the epsilon used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            
        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                state["iteration"] += 1

                mdecay, epsilon, lr = group["mdecay"], group["epsilon"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                gradient = parameter.grad.data * scale_grad

                tau_inv = 1. / (tau + 1.)
                # g = g_theta
                # gradient = delta U(theta)
                # v_hat = V_theta, variance of gradient
                # tau = tau, free parameter vector specifying the exponential averaging windows, automatically
                # update parameters during burn-in
                if state["iteration"] <= group["num_burn_in_steps"]:
                    tau.add_(- tau * (
                            g * g / (v_hat + epsilon)) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    g.add_(-g * tau_inv + tau_inv * gradient)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))  # gradient variance see Eq 8 in [1]

                ## minv_t = M^(-1) = diag(V_theta^(-1/2)), 
                ## maybe according to the v, M^(-1) = sqrt(1/(v_hat))
                minv_t = 1. / (torch.sqrt(v_hat) + epsilon)  # preconditioner
                
                ## equation 10
                ## variance on the random normal
                ## lr = epsilon
                ## mdecay = C
                ## why there is no V_theta^(-1/2)? maybe due to the assumption
                ## Why only use one minv_t? while there are two in the variance
                ## Why the learning rate is only square instead of cubic
                epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

                # sample random normal on the last term
                ## the normal distribution parameter is the mean and variance, therefore need to sqrt
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
                sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)

                # update momentum (Eq 10 right in [1])
                ## where is the v component? v turns out the momentum
                ## there should be one more learning rate in the second term (epsilon*v^(-1/2)Cv)
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                # update theta (Eq 10 left in [1])
                parameter.data.add_(momentum)

        return loss

# Add prior. Note the gradient is computed by: g_prior + N/n sum_i grad_theta_xi see Eq 4
# in Welling and Whye The 2011. Because of that we divide here by N=num of datapoints since
# in the sample we rescale the gradient by N again

'''
prior and noise loss
source: https://github.com/automl/pybnn/blob/master/pybnn/priors.py
'''

def log_variance_prior(log_variance: torch.Tensor, mean: float = 1e-6, variance: float = 0.01) -> torch.Tensor:
    return torch.mean(
        torch.sum(
            ((-((log_variance - torch.log(torch.tensor(mean, dtype=log_variance.dtype))) ** 2)) /
             (2. * variance)) - 0.5 * torch.log(torch.tensor(variance, dtype=log_variance.dtype)),
            dim=1
        )
    )


def weight_prior(parameters: Iterable[torch.Tensor], dtype=np.float64, wdecay: float = 1., device = torch.device('cuda:0')) -> torch.Tensor:

    num_parameters = 0
    log_likelihood = torch.from_numpy(np.array(0, dtype=dtype)).to(device)
    for parameter in parameters:
        num_parameters += parameter.numel()
        log_likelihood += torch.sum(-wdecay * 0.5 * (parameter ** 2))

    return log_likelihood / num_parameters


def get_prior(parameters, dataset_size):
    '''
    prior loss function from 
    https://github.com/ranery/Bayesian-CycleGAN/blob/master/models/CycleGAN_bayes.py#L549
    '''
    prior_loss = Variable(torch.zeros((1))).cuda()
    for param in parameters:
        prior_loss += torch.mean(param*param)
    return prior_loss / dataset_size


'''
prior and noise loss
source: https://github.com/vasiloglou/mltrain-nips-2017/blob/master/ben_athiwaratkun/pytorch-bayesgan/models/bayes.py
'''
class NoiseLoss(torch.nn.Module):
    # need the scale for noise standard deviation
    # scale = noise  std
    def __init__(self, params, scale=None, observed=None):
        super(NoiseLoss, self).__init__()
        # initialize the distribution for each parameter
        #self.distributions = []
        self.noises = []
        for param in params:
            noise = 0*param.data.cuda() # will fill with normal at each forward
            self.noises.append(noise)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1.
        self.observed = observed

    def forward(self, params, scale=None, observed=None):
        # scale should be sqrt(2*alpha/eta)
        # where eta is the learning rate and alpha is the strength of drag term
        if scale is None:
            scale = self.scale
        if observed is None:
            observed = self.observed

        assert scale is not None, "Please provide scale"
        noise_loss = 0.0
        for noise, var in zip(self.noises, params):
            # This is scale * z^T*v
            # The derivative wrt v will become scale*z
            _noise = noise.normal_(0,1)
            noise_loss += scale*torch.sum(Variable(_noise)*var)
        noise_loss /= observed
        return noise_loss

class PriorLoss(torch.nn.Module):
    # negative log Gaussian prior
    def __init__(self, prior_std=1., observed=None):
        super(PriorLoss, self).__init__()
        self.observed = observed
        self.prior_std = prior_std

    def forward(self, params, observed=None):
        if observed is None:
            observed = self.observed
            prior_loss = 0.0
        for var in params:
            prior_loss += torch.sum(var*var/(self.prior_std*self.prior_std))
        prior_loss /= observed
        return prior_loss

# Hessian and Jacobian code from: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


class KFACLaplace(torch.optim.Optimizer):
    r"""KFAC Laplace: based on Scalable Laplace
    Code is partially copied from https://github.com/Thrandis/EKFAC-pytorch/kfac.py.
    TODO: batch norm implementation
    TODO: use some sort of validation set for scaling data_size parameter
    """

    def __init__(
        self,
        net,
        eps,
        sua=False,
        pi=False,
        update_freq=1,
        alpha=1.0,
        constraint_norm=False,
        data_size=50000,
        use_batch_norm=False,
    ):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
            use_batch_norm: whether or not batch norm layers should be computed
        """
        self.net = net
        self.state = net.state_dict()
        self.mean_state = copy.deepcopy(self.state)
        self.data_size = data_size
        self.use_batch_norm = use_batch_norm

        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ["Linear", "Conv2d"]:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {"params": params, "mod": mod, "layer_type": mod_class}
                self.params.append(d)

            elif "BatchNorm" in mod_class and use_batch_norm:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)

                params = [mod.weight, mod.bias]

                d = {"params": params, "mod": mod, "layer_type": mod_class}
                self.params.append(d)

        super(KFACLaplace, self).__init__(self.params, {})
        # super(KFACLaplace, self).__init__()

    def cuda(self):
        self.net.cuda()

    def load_state_dict(self, checkpoint, **kwargs):
        self.net.load_state_dict(checkpoint, **kwargs)

        self.mean_state = self.net.state_dict()

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def apply(self, *args, **kwargs):
        self.net.apply(*args, **kwargs)

    def sample(self, scale=1.0, **kwargs):

        for group in self.params:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None
            state = self.state[weight]

            if "BatchNorm" in group["layer_type"] and self.use_batch_norm:

                z = torch.zeros_like(weight).normal_()
                sample = state["w_ic"].matmul(z)

                if bias is not None:

                    z = torch.zeros_like(bias).normal_()
                    bias_sample = state["b_ic"].matmul(z)

            else:
                # now compute inverse covariances
                # self._compute_covs(group, state)
                ixxt, iggt, ixxt_chol, iggt_chol = self._inv_covs(
                    state["xxt"], state["ggt"], num_locations=state["num_locations"]
                )
                state["ixxt"] = ixxt
                state["iggt"] = iggt

                # draw samples from AZB
                # appendix B of ritter et al.
                z = torch.randn(
                    state["ixxt"].size(0),
                    state["iggt"].size(0),
                    device=ixxt.device,
                    dtype=ixxt.dtype,
                )
                # z = torch.randn(state['ixxt'].size(0), state['iggt'].size(0), dtype = ixxt.dtype)
                # matmul a z b
                # print(state['ixxt'].shape, state['iggt'].shape)
                sample = ixxt_chol.matmul(z.matmul(iggt_chol)).t()
                # sample = ixxt_chol.cpu().matmul(z.matmul(iggt_chol.cpu())).t()
                sample *= scale / self.data_size  # scale/N term for inverse
                # sample = sample.cuda()

                if bias is not None:
                    # print(weight.shape, bias.shape, sample.shape)
                    bias_sample = sample[:, -1].contiguous().view(*bias.shape)
                    sample = sample[:, :-1]
                    # print(weight.shape, bias.shape, sample.shape)

            # print(weight.norm(), sample.norm())
            # finally update parameters with new values as mean is current state dict
            weight.data.add_(sample.view_as(weight))
            if bias is not None:
                bias.data.add_(bias_sample.view_as(bias))

    def step(self, update_stats=True, update_params=True):
        # Performs one step of preconditioning.
        fisher_norm = 0.0
        for group in self.param_groups:
            # print(torch.cuda.memory_allocated()/(1024**3))
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None
            state = self.state[weight]

            # print(group['layer_type'])
            if "BatchNorm" in group["layer_type"] and self.use_batch_norm:
                # now compute hessian of weights
                diag_comp = (
                    100
                    * weight.size(0)
                    * self.eps
                    * torch.eye(
                        weight.size(0), device=weight.device, dtype=weight.dtype
                    )
                )
                # print(weight.size())
                weight_hessian = jacobian(weight.grad, weight) + diag_comp
                # print(weight_hessian)

                weight_inv_chol = torch.cholesky(weight_hessian)
                state["w_ic"] = weight_inv_chol

                if bias is not None:
                    diag_comp = (
                        100
                        * self.eps
                        * torch.eye(bias.size(0), device=bias.device, dtype=bias.dtype)
                    )
                    bias_hessian = jacobian(bias.grad, bias) + diag_comp

                    state["b_ic"] = torch.cholesky(bias_hessian)

            if group["layer_type"] in ["Linear", "Conv2d"]:

                # Update convariances and inverses
                if update_stats:
                    if self._iteration_counter % self.update_freq == 0:
                        self._compute_covs(group, state)
                        ixxt, iggt, _, _ = self._inv_covs(
                            state["xxt"], state["ggt"], state["num_locations"]
                        )
                        state["ixxt"] = ixxt
                        state["iggt"] = iggt
                    else:
                        if self.alpha != 1:
                            self._compute_covs(group, state)
                if update_params:
                    # Preconditionning
                    gw, gb = self._precond(weight, bias, group, state)
                    # Updating gradients
                    if self.constraint_norm:
                        fisher_norm += (weight.grad * gw).sum()
                    weight.grad.data = gw
                    if bias is not None:
                        if self.constraint_norm:
                            fisher_norm += (bias.grad * gb).sum()
                        bias.grad.data = gb
                # Cleaning
                if "x" in self.state[group["mod"]]:
                    del self.state[group["mod"]]["x"]
                if "gy" in self.state[group["mod"]]:
                    del self.state[group["mod"]]["gy"]
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            f_scale = (1.0 / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group["params"]:
                    param.grad.data *= f_scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]["x"] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]["gy"] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group["layer_type"] == "Conv2d" and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        if group["layer_type"] == "Conv2d":
            g = g.contiguous().view(s[0], s[1] * s[2] * s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group["layer_type"] == "Conv2d":
            g /= state["num_locations"]
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        mod = group["mod"]
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0] * s[2] * s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state["num_locations"]
        if bias is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group["mod"]
        x = self.state[group["mod"]]["x"]
        gy = self.state[group["mod"]]["gy"]
        # Computation of xxt
        if group["layer_type"] == "Conv2d":
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state["xxt"] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state["xxt"].addmm_(
                mat1=x,
                mat2=x.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(x.shape[1]),
            )
        # Computation of ggt
        if group["layer_type"] == "Conv2d":
            gy = gy.data.permute(1, 0, 2, 3)
            state["num_locations"] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state["num_locations"] = 1
        if self._iteration_counter == 0:
            state["ggt"] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state["ggt"].addmm_(
                mat1=gy,
                mat2=gy.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(gy.shape[1]),
            )

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = tx / tg
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)

        # Compute cholesky
        xxt_chol = (xxt + torch.diag(diag_xxt)).cholesky()
        ggt_chol = (ggt + torch.diag(diag_ggt)).cholesky()

        # invert cholesky
        xxt_ichol = torch.inverse(xxt_chol)
        ggt_ichol = torch.inverse(ggt_chol)

        # invert matrix
        ixxt = xxt_ichol.t().matmul(xxt_ichol)
        iggt = ggt_ichol.t().matmul(ggt_ichol)

        return ixxt, iggt, xxt_ichol, ggt_ichol
    
"""
Two coin betting optimization algorithms are implemented here :
Cocob Backprop: https://arxiv.org/pdf/1705.07795.pdf
Cocob through Ons: https://arxiv.org/pdf/1705.07795.pdf
both of which do not require any learning rates and yet
have optimal convergence gauarantees for non-smooth
convex functions.

Cocob-Ons is an experimental variation from paper.
Please don't use it yet.

Please check http://francesco.orabona.com/papers/slides_cocob.pdf for
simple explanation for going from coin betting game to convex optimization.
Both algorithms are similar except the coin betting strategy used.
"""

class CocobBackprop(optim.Optimizer):
    """Implements Cocob-Backprop .

    It has been proposed in `Training Deep Networks without Learning Rates
    Through Coin Betting`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): positive number to adjust betting fraction.
            Theoretical convergence gauarantee does not depend on choice of
            alpha (default: 100.0)

    __ https://arxiv.org/pdf/1705.07795.pdf
    """
    def __init__(self, params, alpha=100.0, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        defaults = dict(alpha=alpha, eps=eps)
        super(CocobBackprop, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                param_shape = param.shape

                # Better bets for -ve gradient
                neg_grad = - grad

                if len(state) == 0:
                    # Happens only once at the begining of optimization start
                    # Set initial parameter weights and zero reward
                    state['initial_weight'] = param.data
                    state['reward'] = param.new_zeros(param_shape)

                    # Don't bet anything for first round
                    state['bet'] = param.new_zeros(param_shape)

                    # Initialize internal states useful for computing betting fraction
                    state['neg_grads_sum'] = param.new_zeros(param_shape)
                    state['grads_abs_sum'] = param.new_zeros(param_shape)
                    state['max_observed_scale'] = self.eps * param.new_ones(param_shape)

                # load states in variables
                initial_weight = state['initial_weight']
                reward = state['reward']
                bet = state['bet']
                neg_grads_sum = state['neg_grads_sum']
                grads_abs_sum = state['grads_abs_sum']
                max_observed_scale = state['max_observed_scale']

                # Update internal states useful for computing betting fraction
                max_observed_scale = torch.max(max_observed_scale, torch.abs(grad))
                grads_abs_sum += torch.abs(grad)
                neg_grads_sum += neg_grad

                # Based on how much the Better bets on -ve gradient prediction,
                # check how much the Better won (-ve if lost)
                win_amount = bet * neg_grad

                # Update better's reward. Negative reward is not allowed.
                reward = torch.max(reward + win_amount, torch.zeros_like(reward))

                # Better decides the bet fraction based on so-far observations
                bet_fraction = neg_grads_sum / (max_observed_scale * (torch.max(grads_abs_sum + max_observed_scale, self.alpha * max_observed_scale)))

                # Better makes the bet according to decided betting fraction.
                bet = bet_fraction * (max_observed_scale + reward)

                # Set parameter weights
                param.data = initial_weight + bet

                # save state back in memory
                state['neg_grads_sum'] = neg_grads_sum
                state['grads_abs_sum'] = grads_abs_sum
                state['max_observed_scale'] = max_observed_scale
                state['reward'] = reward
                state['bet'] = bet
                # For Cocob-Backprop bet_fraction need not be maintained in state. Only kept for visualization.
                state['bet_fraction'] = bet_fraction

        return loss


class CocobOns(optim.Optimizer):
    """Implements Coin-Betting through ONS .

    It has been proposed in `Black-Box Reductions for Parameter-free
    Online Learning in Banach Spaces`__.

    Cocob-Ons is an experimental variation from the paper.
    Do not use it yet.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eps (float, optional): positive initial wealth for betting algorithm.
            Theoretical convergence gauarantee does not depend on choice of
            eps (default: 1e-8)

    __ https://arxiv.org/pdf/1705.07795.pdf
    """
    def __init__(self, params, eps=1e-8):

        self.eps = eps
        defaults = dict(eps=eps)
        super(CocobOns, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                param_shape = param.data.shape

                # Clip gradients to be in (-1, 1)
                grad.clamp_(-1.0, 1.0)

                # Better bets for -ve gradient
                neg_grad = - grad

                if len(state) == 0:
                    # Happens only once at the begining of optimization start
                    # Set initial parameter weights and zero reward
                    state['initial_weight'] = param.data
                    state['wealth'] = self.eps * param.new_ones(param_shape)

                    # Don't bet anything for first round
                    state['bet_fraction'] = param.new_zeros(param_shape)
                    state['bet'] = param.new_zeros(param_shape)

                    # Initialize internal states useful for computing betting fraction
                    state['z_square_sum'] = param.new_zeros(param_shape)

                # load states in memory
                wealth = state['wealth']
                bet_fraction = state['bet_fraction']
                z_square_sum = state['z_square_sum']
                initial_weight = state['initial_weight']
                bet = state['bet']

                # Based on how much the Better bets on -ve gradient prediction,
                # check how much the Better won (-ve if lost)
                win_amount = bet * neg_grad

                # Update better's wealth based on what he won / lost.
                wealth = wealth + win_amount

                # Better decides the bet fraction based on so-far observations
                # z, A variable notations from Algo 1 in paper)
                z = grad / (1-(bet_fraction*grad))
                z_square_sum = z_square_sum + (z*z)
                A = 1 + z_square_sum

                bet_fraction = (bet_fraction - (2/(2 - math.log(3)))*(z / A))
                bet_fraction.clamp_(-0.5, 0.5)

                # Better makes the bet according to decided betting fraction.
                bet = bet_fraction * wealth

                # Set parameter weights
                param.data = initial_weight + bet

                # save state back in memory
                state['bet_fraction'] = bet_fraction
                state['wealth'] = wealth
                state['z_square_sum'] = z_square_sum
                state['bet'] = bet

        return loss