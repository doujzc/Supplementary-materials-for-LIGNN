import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LILayer(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim=None, bias=True, tol=1e-2):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if out_dim == None:
            out_dim = 1
            self.squeeze_out = True
        else:
            self.squeeze_out = False
        self.out_dim = out_dim

        self.inlin = Linear(in_dim, hidden_dim, bias=False)
        self.outlin = Linear(hidden_dim, out_dim, bias=True)
        self.W = Linear(in_dim, hidden_dim*hidden_dim, bias=bias)
        self.solver = self.forward_iteration
        self.tol = tol

    def forward_iteration(self, f, x0, max_iter=50):
        f0 = f(x0)
        res = []
        for k in range(max_iter):
            x = f0
            f0 = f(x)
            res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
            if (res[-1] < self.tol):
                break
        return f0

    def forward(self, t, edge_index, edge_attr, N, max_iteration=5, contract=0.8):
        ''' Apply forward iteration.
        t: [bsz]
        edge_index: [2, E]
        edge_attr: [E, in_dim]
        N: number of nodes
        '''
        bsz = t.shape[0]
        E = edge_index.shape[1]

        # Step 1: Build initial x_st.
        # x0: [N, bsz, hidden]
        x00 = self.inlin(edge_attr)
        x0 = torch.zeros((N, bsz, self.hidden_dim), device=edge_index.device)
        st, ed = edge_index
        batch, edge_id = (st==t.view(-1,1)).nonzero().T
        ed_batch = ed[edge_id]
        x0[ed_batch, batch] = x00[edge_id]

        # Step 2: Build initial X_st.
        # X_st: [E, hidden, hidden]
        dout = degree(edge_index[0], N)
        W = self.W(edge_attr)
        W = W / (dout[st] * W.norm(p='fro',dim=-1) + 1e-3).view(-1, 1)
        W = W.view(E, self.hidden_dim, self.hidden_dim)

        # Step 3: Start propagating messages.
        def f(xx):
            xx = xx.view(N, -1)
            x1 = self.propagate(edge_index=edge_index, size=(N,N), x=xx, W=W).view_as(x0)
            return contract * x1 + x0

        with torch.no_grad():
            z = self.solver(f, x0, max_iter=max_iteration)
        z = f(z)

        z0 = z.clone().detach().requires_grad_()
        f0 = f(z0)
        def backward_hook(grad):
            g = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad,
                                                    max_iteration)
            return g
        if z.requires_grad:
            z.register_hook(backward_hook)

        # Step 4: Compute outputs.
        out = z.view(N, bsz, self.hidden_dim).permute(1,0,2)
        ret = self.outlin(out)
        if self.squeeze_out:
            return ret.squeeze()
        else:
            return ret

    def message(self, x_j, W):
        # x_j [E, bsz*hidden]
        # W [E, hidden, hidden]
        x_j = x_j.view(x_j.shape[0], -1, self.hidden_dim)
        x_j = (W.unsqueeze(1) @ x_j.unsqueeze(-1)).view(x_j.shape[0], -1)
        return x_j

    def rescale(self):
        with torch.no_grad():
            self.W.weight /= self.W.weight.norm(dim=1).view(-1,1)



class LILayer_ty(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim=None, bias=True, tol=1e-2):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if out_dim == None:
            out_dim = 1
            self.squeeze_out = True
        else:
            self.squeeze_out = False
        self.out_dim = out_dim

        self.x0 = Parameter(torch.randn(in_dim, hidden_dim))
        # self.x0 = torch.eye(in_dim)
        self.W = Parameter(torch.randn(in_dim, hidden_dim, hidden_dim))
        # self.register_parameter('x0', x0)
        # self.register_parameter('W', W)
        self.outlin = Linear(hidden_dim, out_dim, bias=bias)
        self.solver = self.forward_iteration
        self.tol = tol

    def forward_iteration(self, f, x0, max_iter=50):
        f0 = f(x0)
        res = []
        for _ in range(max_iter):
            x = f0
            f0 = f(x)
            res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
            if (res[-1] < self.tol):
                break
        return f0

    def forward(self, s, edge_index, edge_type, N, max_iteration=5, contract=0.8):
        ''' Apply forward iteration.
        s: [bsz]
        edge_index: [2, E]
        edge_attr: [E]
        N: number of nodes
        '''
        bsz = s.shape[0]
        E = edge_index.shape[1]

        # Step 1: Build initial x_st.
        # x0: [N, bsz, hidden]
        x00 = self.x0[edge_type]
        x0 = torch.zeros((N, bsz, self.hidden_dim), device=edge_index.device)
        st, ed = edge_index
        batch, edge_id = (st==s.view(-1,1)).nonzero().T
        ed_batch = ed[edge_id]
        x0[ed_batch, batch] = x00[edge_id]

        # Step 2: Build initial X_st.
        # W: [E, hidden, hidden]
        dout = degree(edge_index[0], N)
        W = self.W[edge_type]
        W = W / (dout[st] * W.norm(p='fro', dim=[1,2]) + 1e-3).view(-1, 1, 1)
        # din = degree(edge_index[1], N)
        # W = self.W[edge_type]
        # W = W / (din[ed] * W.norm(p='fro', dim=[1,2]) + 1e-3).view(-1, 1, 1)
        # W = W / (din[ed] * W.norm(dim=-1) + 1e-3).view(-1, 1)
        # W = W / (din[ed] * self.hidden_dim + 1e-3).view(-1, 1)
        # W = W.view(E, self.hidden_dim, self.hidden_dim)

        # Step 3: Start propagating messages.
        def f(xx):
            xx = xx.view(N, -1)
            x1 = self.propagate(edge_index=edge_index, size=(N,N), x=xx, W=W).view_as(x0)
            return contract * x1 + x0

        with torch.no_grad():
            z = self.solver(f, x0, max_iter=max_iteration)
        z = f(z)

        z0 = z.clone().detach().requires_grad_()
        f0 = f(z0)
        def backward_hook(grad):
            g = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad,
                                                    max_iteration)
            return g
        if z.requires_grad:
            z.register_hook(backward_hook)

        # Step 4: Compute outputs.
        # z = z / (1e-2+z.norm(dim=-1).view(N,bsz,1))
        out = z.view(N, bsz, self.hidden_dim).permute(1,0,2)
        ret = self.outlin(out)# + self.bias
        # ret = self.outlin(torch.relu(self.out1(torch.relu(self.out2(torch.sigmoid(out)))))) + self.bias
        if self.squeeze_out:
            return ret.squeeze()
        else:
            return ret

    def message(self, x_j, W):
        # x_j [E, bsz*hidden]
        # W [E, hidden, hidden]
        x_j = x_j.view(x_j.shape[0], -1, self.hidden_dim)
        x_j = (W.unsqueeze(1) @ x_j.unsqueeze(-1)).view(x_j.shape[0], -1)
        return x_j

    def rescale(self):
        with torch.no_grad():
            self.W /= self.W.norm(dim=0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1:
                torch.nn.init.xavier_uniform_(p)