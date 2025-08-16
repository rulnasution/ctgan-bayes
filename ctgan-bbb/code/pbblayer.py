import warnings, math, torch, numpy as np
from torch.nn import init, Module, Parameter, functional as F, _reduction as _Reduction, Sigmoid

### modified from https://github.com/Harry24k/bayesian-neural-network-pytorch

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
    
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class BayesLinear(Module):
    r"""
    Applies Bayesian Linear

    Arguments:
        prior_dist (string): prior distribution used 
            'gauss' for gaussian, for both exp (see adv-bnn) with softplus (see blundell et al.)
            'laplace': for laplace
        std_act (string): activation for sigma for gaussian or b1/scale for laplace  
            'exp', using exp (rho) (see adv-bnn)
            'softplus': with softplus function log(1+exp(rho)) (see blundell et al.)
        prior_mu (Float): mean of prior normal distribution and laplace.
        prior_sigma (Float): sigma of prior normal distribution or scale for laplace.
        

    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_dist, std_act, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_dist = prior_dist
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        # print(prior_mu,prior_sigma)
        self.std_act = std_act
        if std_act == 'exp':
            self.prior_log_sigma = math.log(prior_sigma)
        elif std_act == 'softplus':
            self.prior_log_sigma = math.log(math.exp(prior_sigma)-1)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))

        # print('a')
        # print(self.prior_mu, self.prior_log_sigma)
        # print('b')
        # print(self.weight_mu, self.weight_log_sigma)
        self.register_buffer('weight_eps', None)
                
        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True
            
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        if self.prior_dist == 'gauss':
            self.weight_eps = torch.randn_like(self.weight_log_sigma)
            if self.bias :
                self.bias_eps = torch.randn_like(self.bias_log_sigma)
        elif self.prior_dist == 'laplace':
            self.weight_eps = (0.999*torch.rand(self.weight_log_sigma.size())-0.49999)
            if self.bias :
                self.bias_eps = (0.999*torch.rand(self.bias_log_sigma.size())-0.49999)
                
    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None 
            
    def forward(self, input):
        r"""
        Overriden.
        """

        # with torch.no_grad():
        #     if self.std_act == 'exp':
        #         weight_sigma = torch.exp(self.weight_log_sigma)
        #     elif self.std_act == 'softplus':
        #         weight_sigma = torch.log(1+torch.exp(self.weight_log_sigma))
                    # print( epsilon.get_device())
        device = torch.device('cpu') if self.bias_log_sigma.get_device() == -1 else torch.device('cuda')
        if self.weight_eps is None :
            if self.prior_dist == 'gauss':
                if self.std_act == 'exp':
                    weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
                elif self.std_act == 'softplus':
                    weight = self.weight_mu + torch.log(1+torch.exp(self.weight_log_sigma)) * torch.randn_like(self.weight_log_sigma)
            elif self.prior_dist == 'laplace':
                epsilon = (0.999*torch.rand(self.weight_log_sigma.size())-0.49999)
                epsilon = epsilon.to(device)
                if self.std_act == 'exp':
                    p1 = torch.exp(self.weight_log_sigma) * torch.sign(epsilon)
                elif self.std_act == 'softplus':
                    p1 = torch.log(1+torch.exp(self.weight_log_sigma)) * torch.sign(epsilon)
                weight = self.weight_mu - p1 * torch.log(1-2*torch.abs(epsilon))
        else :
            if self.prior_dist == 'gauss':
                if self.std_act == 'exp':
                    weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
                elif self.std_act == 'softplus':
                    weight = self.weight_mu + torch.log(1+torch.exp(self.weight_log_sigma)) * self.weight_eps
            elif self.prior_dist == 'laplace':
                epsilon = (0.999*torch.rand(self.weight_log_sigma.size())-0.49999)
                epsilon = epsilon.to(device)
                if self.std_act == 'exp':
                    p1 = torch.exp(self.weight_log_sigma) * torch.sign(self.weight_eps)
                elif self.std_act == 'softplus':
                    p1 = torch.log(1+torch.exp(self.weight_log_sigma)) * torch.sign(self.weight_eps)
                weight = self.weight_mu - p1 * torch.log(1-2*torch.abs(self.weight_eps))
        
        if self.bias:
            # with torch.no_grad():
            #     if self.std_act == 'exp':
            #         bias_sigma = torch.exp(self.bias_log_sigma)
            #     elif self.std_act == 'softplus':
            #         bias_sigma = torch.log(1+torch.exp(self.bias_log_sigma))

            if self.bias_eps is None :
                if self.prior_dist == 'gauss':
                    if self.std_act == 'exp':
                        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
                    elif self.std_act == 'softplus':
                        bias = self.bias_mu + torch.log(1+torch.exp(self.bias_log_sigma)) * torch.randn_like(self.bias_log_sigma)
                elif self.prior_dist == 'laplace':
                    epsilon = (0.999*torch.rand(self.bias_log_sigma.size())-0.49999)
                    epsilon = epsilon.to(device)
                    if self.std_act == 'exp':
                        p1 = torch.exp(self.bias_log_sigma) * torch.sign(epsilon)
                    elif self.std_act == 'softplus':
                        p1 = torch.log(1+torch.exp(self.bias_log_sigma)) * torch.sign(epsilon)
                    bias = self.bias_mu - p1 * torch.log(1-2*torch.abs(epsilon))

            else :
                if self.prior_dist == 'gauss':
                    if self.std_act == 'exp':
                        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
                    elif self.std_act == 'softplus':
                        bias = self.bias_mu + torch.log(1+torch.exp(self.bias_log_sigma)) * self.bias_eps
                elif self.prior_dist == 'laplace':
                    epsilon = (0.999*torch.rand(self.bias_log_sigma.size())-0.49999)
                    epsilon = epsilon.to(device)
                    if self.std_act == 'exp':
                        p1 = torch.exp(self.bias_log_sigma) * torch.sign(self.bias_eps)
                    elif self.std_act == 'softplus':
                        p1 = torch.log(1+torch.exp(self.bias_log_sigma)) * torch.sign(self.bias_eps)
                    bias = self.bias_mu - p1 * torch.log(1-2*torch.abs(self.bias_eps))

            # if self.bias_eps is None :
            #     if self.prior_dist == 'gauss':
            #         bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            #     elif self.prior_dist == 'laplace':
            #         epsilon = (0.999*torch.rand(bias_sigma.size())-0.49999)
            #         p1 = bias_sigma * torch.sign(epsilon)
            #         bias = self.bias_mu - p1 * torch.log(1-2*torch.abs(epsilon))

            #     else :
            #         if self.prior_dist == 'gauss':
            #             bias = self.bias_mu + bias_sigma * self.bias_eps
            #         elif self.prior_dist == 'laplace':
            #             p1 = bias_sigma * torch.sign(self.bias_eps)
            #             bias = self.bias_mu - p1 * torch.log(1-2*torch.abs(self.bias_eps))
             
        else :
            bias = None

        # print(self.prior_dist, self.std_act, self.prior_mu, self.prior_sigma)
        
        # print(input.dtype, weight.dtype, bias.dtype)
        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)
    

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1, prior_dist = 'gauss', std_act = 'exp') :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
        prior_dist (string): prior distribution used 
            'gauss' for gaussian, for both exp (see adv-bnn) with softplus (see blundell et al.)
            'laplace': for laplace
        std_act (string): activation for sigma for gaussian or b1/scale for laplace  
            'exp', using exp (rho) (see adv-bnn)
            'softplus': with softplus function log(1+exp(rho)) (see blundell et al.)
    """
    if std_act == 'exp':
        sigma_0 = torch.exp(log_sigma_0)
        sigma_1 = math.exp(log_sigma_1)
    elif std_act == 'softplus':
        sigma_0 = torch.log(1.0 + torch.exp(log_sigma_0))
        sigma_1 = math.log(1.0 + math.exp(log_sigma_1))

    if prior_dist=='gauss':
        kl = log_sigma_1 - log_sigma_0 + \
        (sigma_0**2 + (mu_0-mu_1)**2)/(2*sigma_1**2) - 0.5
    elif prior_dist == 'laplace':
        '''
         adopted from 
         https://openaccess.thecvf.com/content/CVPR2021/supplemental/
         Meyer_An_Alternative_Probabilistic_CVPR_2021_supplemental.pdf
        '''
        p1 = torch.abs(mu_0-mu_1)
        p2 = torch.exp(-p1/sigma_0)
        p3 = torch.log(sigma_1/sigma_0)
        pp1 = (sigma_1*p2 + p1)/sigma_1
        kl =  pp1 + p3 - 1.0
    return kl.sum()


class Lambda_var(Module):
    """Class for the lambda variable included in the objective
    flambda

    Parameters
    ----------
    lamb : float
        initial value, equal to -> initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)

    n : int
        Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)
        equal to train_size = len(train_loader.dataset) ???

    """

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1/np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = Sigmoid()
        return (m(self.lamb) * (1-self.min) + self.min)
    
def bayesian_kl_loss(model, prior_dist = 'gauss', std_act = 'exp', reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.

    The layer should be BayesLinear
    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        prior_dist (string): prior distribution used 
            'gauss' for gaussian, for both exp (see adv-bnn) with softplus (see blundell et al.)
            'laplace': for laplace
        std_act (string): activation for sigma for gaussian or b1/scale for laplace  
            'exp', using exp (rho) (see adv-bnn)
            'softplus': with softplus function log(1+exp(rho)) (see blundell et al.)


        last_layer_only (Bool): True for return only the last layer's KL divergence.    
        
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, (BayesLinear)):
            # print(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma, prior_dist, std_act)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma, prior_dist, std_act)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))
                
    if last_layer_only or n == 0 :
        return kl
    
    if reduction == 'mean' :
        return kl_sum/n
    elif reduction == 'sum' :
        return kl_sum
    else :
        raise ValueError(reduction + " is not valid")
    
class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            
class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        prior_dist (string): prior distribution used 
            'gauss' for gaussian, for both exp (see adv-bnn) with softplus (see blundell et al.)
            'laplace': for laplace
        std_act (string): activation for sigma for gaussian or b1/scale for laplace  
            'exp', using exp (rho) (see adv-bnn)
            'softplus': with softplus function log(1+exp(rho)) (see blundell et al.)
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, prior_dist = 'gauss', std_act = 'exp', reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__()
        self.prior_dist = prior_dist 
        self.std_act = std_act
        self.reduction = reduction
        
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return bayesian_kl_loss(model, prior_dist = self.prior_dist, std_act = self.std_act, 
                                reduction=self.reduction, last_layer_only=self.last_layer_only)


def bayes_total_loss(empirical_risk, kl, train_size, lambda_var=None,
          objective='fquad', kl_penalty=1, delta=0.025
    ):
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired 
    training objective and evaluate the risk certificate at the end of training. 

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)
    
    delta : float
        confidence value for the training objective

    kl_penalty : float
        penalty for the kl coefficient in the training objective
    
    device : string
        Device the code will run in (e.g. 'cuda')

    """
    # compute training objectives
    if objective == 'fquad':
        kl = kl * kl_penalty
        repeated_kl_ratio = torch.div(
            kl + torch.log((2*torch.sqrt(train_size))/delta), 2*train_size)
        first_term = torch.sqrt(
            empirical_risk + repeated_kl_ratio)
        second_term = torch.sqrt(repeated_kl_ratio)
        train_obj = torch.pow(first_term + second_term, 2)
    elif objective == 'flamb':
        kl = kl * kl_penalty
        lamb = lambda_var.lamb_scaled
        kl_term = torch.div(
            kl + torch.log((2*torch.sqrt(train_size)) / delta), train_size*lamb*(1 - lamb/2))
        first_term = torch.div(empirical_risk, 1 - lamb/2)
        train_obj = first_term + kl_term
    elif objective == 'fclassic':
        kl = kl * kl_penalty
        kl_ratio = torch.div(
            kl + torch.log((2*torch.sqrt(train_size))/delta), 2*train_size)
        train_obj = empirical_risk + torch.sqrt(kl_ratio)
    elif objective == 'bbb':
        # ipdb.set_trace()
        train_obj = empirical_risk + \
            kl_penalty * (kl/train_size)
    else:
        raise RuntimeError(f'Wrong objective {objective}')
    return train_obj
