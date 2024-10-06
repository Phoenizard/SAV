import math
import torch
from torch.optim import Optimizer

class SAV(Optimizer):
    def __init__(self, params, lr=1, const=2e-2, weight_decay=0):
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, const=const, weight_decay=weight_decay)
        super(SAV, self).__init__(params, defaults)
    
    def _Ainv(self, x, alpha):            
        return x/(1+alpha)
    
    def _L(self, group, lambda_l2):
        res = 0.
        for p in group['params']:
            if p is None:
                continue
            res += (torch.norm(p)**2).item()
#        print(0.5*lambda_l2*res)
        return 0.5 * lambda_l2 * res
    
    def _N(self, group, lambda_l2, loss, const):
        L = self._L(group, lambda_l2)
        N = loss - L 
        if N + const < 0.:
            raise ValueError("loss_N + const = {}, please choose a const s.t. N(x)+const > 0".format(N+const))
        return N + const
    
    def _dL(self, x, lambda_l2):
        return lambda_l2*x
    
    def _dN(self, x, grad, lambda_l2):
        dL = self._dL(x, lambda_l2)
        dN = grad - dL
        return dN
    
    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        closure = torch.enable_grad()(closure)
        loss = closure()
        
        for group in self.param_groups:
            lambda_l2 = group['weight_decay']
            alpha = group['lr'] * group['weight_decay']
            loss_N = self._N(group, lambda_l2, loss.item(), group['const'])
#            print(loss_N)
            loss_sqrt = math.sqrt(loss_N)
            
            if not 0.0 < loss_N:
                raise ValueError("loss_N + const = {}, please change the const such that the sum is greater than zero.".format(loss_N))

            bAinvx_sum = 0.
            bAinvb_sum = 0.

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SAV does not support sparse gradients')
                dN = self._dN(p, grad, lambda_l2)
#                print(dN.shape)
#                print(p.shape)
                bAinvx_sum += (torch.sum(torch.mul(dN, (self._Ainv(p,alpha) - p)))).item()
                bAinvb_sum += (torch.sum(torch.mul(dN, (self._Ainv(dN,alpha))))).item()
            bAinvx_sum /= loss_sqrt
            bAinvb_sum /= loss_N
#            print(bAinvx_sum)
#            print(bAinvb_sum)

    #         # Lazy state initialization
    #         if len(state) == 0:                        

            r = loss_sqrt # restart r as a scalar
            r += 0.5 * bAinvx_sum
            r /= (1 + (0.5*group['lr']) * bAinvb_sum)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                p.add_(grad, alpha=-group['lr']*r/loss_sqrt)
                p.div_(1+alpha)
        return loss
