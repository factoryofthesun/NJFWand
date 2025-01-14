import numpy
import igl
import numpy as np
import torch
import time

from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
from scipy import sparse


USE_TORCH_SPARSE = True ## This uses TORCH_SPARSE instead of TORCH.SPARSE

# This four are mutually exclusive
USE_CUPY = False  ## This uses CUPY LU decomposition on GPU
USE_CHOLESPY_GPU = True  ## This uses cholesky decomposition on GPU
USE_CHOLESPY_CPU = False  ## This uses cholesky decomposition on CPU
USE_SCIPY = False ## This uses CUPY LU decomposition on CPU



# If USE_SCIPY = True, wether or not to use enhanced backend
USE_SCIKITS_UMFPACK = False  ## This uses UMFPACK backend for scipy instead of naive scipy.

if USE_CHOLESPY_GPU or USE_CHOLESPY_CPU:
    from cholespy import CholeskySolverD, MatrixType

if USE_CUPY and torch.cuda.is_available():
    from cupyx.scipy.sparse.linalg import spsolve_triangular
    from cupyx.scipy.sparse import csr_matrix
    import cupy
    from torch.utils.dlpack import to_dlpack, from_dlpack

if USE_SCIPY:
    if  USE_SCIKITS_UMFPACK:
        # This is a bit slower in practice
        # https://stackoverflow.com/questions/64401503/is-there-a-way-to-further-improve-sparse-solution-times-using-python
        from scikits.umfpack import splu as scipy_splu
    else:
        import scipy.sparse.linalg as lg
        lg.use_solver(useUmfpack=False)
        # Slight performance gain with True
        # conda install -c conda-forge scikit-umfpack
        # forward pass goes from 0.038 to 0.036
        # assumeSortedIndices=True Does not bring any boost
        from scipy.sparse.linalg import splu as scipy_splu
        from scipy.sparse.linalg import spsolve_triangular, spsolve


if USE_TORCH_SPARSE:
    import torch_sparse


USE_UGLY_PATCH_FOR_CUPY_ERROR = False


class SparseMat:
    '''
    Sparse matrix object represented in the COO format
    Refacto : consider killing this object, byproduct of torch_sparse instead of torch.sparse (new feature)
    '''

    @staticmethod
    def from_M(M,ttype):
        return SparseMat(M[0],M[1],M[2],M[3],ttype)

    @staticmethod
    def from_coo(coo,ttype):
        inds = numpy.vstack((coo.row,coo.col))
        return SparseMat(inds,coo.data,coo.shape[0],coo.shape[1],ttype)

    def __init__(self,inds,vals,n,m,ttype):
        self.n = n
        self.m = m
        self.vals = vals
        self.inds = inds
        assert(inds.shape[0] == 2)
        assert(inds.shape[1] == vals.shape[0])
        assert(np.max(inds[0,:]) <= n)
        assert(np.max(inds[1,:] <= m))
        #TODO figure out how to extract the I,J,V,m,n from this, then load a COO mat directly from npz
        #self.coo_mat = coo_matrix((cupy.array(self.vals), (cupy.array(self.inds[0,:]), cupy.array(self.inds[1,:]))))
        self.vals = torch.from_numpy(self.vals).type(ttype).contiguous()
        self.inds = torch.from_numpy(self.inds).type(torch.int64).contiguous()

    def to_coo(self):
        return coo_matrix((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def to_csc(self):
        return sp_csc((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def update_vals(self, new_vals, target_inds, lap_pinned=None, lap_pinned_rows=None, lap_pinned_cols=None):
        """ target_inds: tuple (tensor, tensor) which indexes into soft laplacian corresponding to new_vals """
        # HACK: convert to dense then back to sparse
        densemat = torch.sparse_coo_tensor(self.inds, self.vals.detach(), (self.n, self.m)).to_dense()
        assert len(target_inds[0]) == len(new_vals), f"target_inds {len(target_inds[0])} and new_vals {len(new_vals)} must be the same length"

        # Add back in pinned values to laplacian
        if lap_pinned is not None:
            for i in range(len(lap_pinned)):
                # The index should never be larger than length of array
                pidx = lap_pinned[i]
                assert densemat.shape[0] >= pidx and densemat.shape[1] >= pidx, f"Pinning error: trying to insert at {pidx} for array of size {densemat.shape}"
                densemat = torch.cat([densemat[:pidx, :], torch.from_numpy(lap_pinned_rows[i]).type_as(densemat).to(self.vals.device), densemat[pidx:,:]], dim=0)
                densemat = torch.cat([densemat[:, :pidx], torch.from_numpy(lap_pinned_cols[i]).type_as(densemat).to(self.vals.device), densemat[:,pidx:]], dim=1)

        densemat[target_inds] = new_vals
        # densemat[target_inds[1], target_inds[0]] = new_vals

        densemat.fill_diagonal_(0)
        densemat[range(len(densemat)), range(len(densemat))] = -torch.sum(densemat, dim=1)
        torch.testing.assert_close(torch.sum(densemat, dim=1), torch.zeros(densemat.shape[0], device=densemat.device, dtype=densemat.dtype), atol=1e-4, rtol=1e-4)

        if lap_pinned is not None:
            for pidx in lap_pinned:
                densemat = torch.cat([densemat[:pidx, :], densemat[pidx+1:,:]], dim=0)
                densemat = torch.cat([densemat[:, :pidx], densemat[:,pidx+1:]], dim=1)

        # Convert back to sparse
        sparsemat = densemat.to_sparse()
        self.vals = sparsemat.values()
        self.inds = sparsemat.indices()

    def to_cholesky(self):
        return CholeskySolverD(self.n, self.inds[0,:], self.inds[1,:], self.vals, MatrixType.COO)

    def to(self,device):
        self.vals = self.vals.to(device)
        self.inds = self.inds.to(device)
        return self

    def pin_memory(self):
        return
        # self.vals.pin_memory()
        # self.inds.pin_memory()

    def multiply_with_dense(self,dense):
        if USE_TORCH_SPARSE:
            res = torch_sparse.spmm(self.inds,self.vals, self.n, self.m, dense)
            # 1000 for loop on the above line takes 0.13 sec. Fast but annoying to have this dependency
        else:
            # Somehow this is not implemented for now?
            # res = torch.smm(torch.sparse_coo_tensor(self.inds,self.vals) , (dense.float())).to_dense().to(dense.device)
            # 1000 for loop on the above line takes 10 sec on the CPU. It is not implemented on gpu yet Slower but no dependency
            if self.vals.device.type == 'cpu':
                tensor_zero_hack  = torch.FloatTensor([0]).double() # This line was somehow responsible for a nasty NAN bug
            else:
                tensor_zero_hack  =  torch.cuda.FloatTensor([0]).to(dense.get_device()).double()
            # beware with addmm, it is experimental and gave me a NaN bug!
            res = torch.sparse.addmm(tensor_zero_hack, torch.sparse_coo_tensor(self.inds.double(),self.vals.double()) , (dense.double())).type_as(self.vals)
            # 1000 for loop on the above line takes 0.77 sec. Slower but no dependency
        return res.contiguous()



class PoissonSystemMatrices:
    '''
    Holds the matrices needed to perform gradient and poisson computations
    Logic : this class is supposed is supposed to hold everything needed to compute Poisson Solver
    Refacto : merge with Poisson Solver
    Only accept SparseMat representation
    '''
    def __init__(self, V, F,grad, rhs, w, ttype, is_sparse = True, lap = None, lap_pinned=None, components=None, cpuonly=False,
                 lap_pinned_rows = None, lap_pinned_cols=None, soft = False, sparse=True):
        self.dim = 3
        self.is_sparse = is_sparse
        self.w = w
        self.rhs = rhs
        self.igl_grad = grad
        self.ttype = ttype
        self.__splu_L = None
        self.__splu_U = None
        self.__splu_perm_c = None
        self.__splu_perm_r = None
        self.lap = lap
        self.lap_pinned = lap_pinned
        self.lap_pinned_rows = lap_pinned_rows
        self.lap_pinned_cols = lap_pinned_cols
        self.components = components
        self.__V = V
        self.__F = F
        self.cpuonly = cpuonly
        self.cpu_splu = None
        self.soft = soft
        self.sparse = sparse

    # NOTE: Creates the poisson solver -- never initializes with my_splu!
    def create_poisson_solver(self):
        return PoissonSolver(self.igl_grad,self.w,self.rhs, None, self.lap, self.lap_pinned, self.components,
                             lap_pinned_rows = self.lap_pinned_rows, lap_pinned_cols=self.lap_pinned_cols,
                             soft = self.soft, sparse=self.sparse)

    def create_poisson_solver_from_splu_old(self, lap_L, lap_U, lap_perm_c, lap_perm_r):
        w = torch.from_numpy(self.w).type(self.ttype)
        lap = None
        my_splu = None
        if not self.cpuonly:
            if USE_CUPY:
                my_splu = MyCuSPLU(lap_L, lap_U, lap_perm_c, lap_perm_r)
            else:
                if self.lap is not None:
                    lap = self.lap
                    # my_splu = scipy_splu(self.lap)
                    # my_splu = MyCuSPLU_CPU(lap_L, lap_U, lap_perm_c, lap_perm_r)
                else:
                    my_splu = MyCuSPLU_CPU(lap_L, lap_U, lap_perm_c, lap_perm_r)
                # st = time.time()
                # my_splu = scipy_splu(lap_L@lap_U)
                # print(f"time for LU: {time.time() - st}" )

        else:
            if self.lap is not None:
                my_splu = scipy_splu(self.lap)
            else:
                0/0
                # my_splu = splu(lap_L)

        return PoissonSolver(self.igl_grad,w,self.rhs,my_splu, lap, sparse=self.sparse)

    def compute_poisson_solver_from_laplacian(self, compute_splu=True):
        self.compute_laplacian()
        if compute_splu:
            self.compute_splu()
        return self.create_poisson_solver_from_splu(self.__splu_L,self.__splu_U,self.__splu_perm_c,self.__splu_perm_r)

    def compute_laplacian(self):
        if self.lap is None:
            self.lap = igl.cotmatrix(self.__V,self.__F)
            self.lap = self.lap[1:, 1:] # NOTE: This is where the pinning alters the laplacian
            self.lap = SparseMat.from_coo(self.lap.tocoo(), torch.float64)

        if isinstance(self.lap,PoissonSystemMatrices) and self.lap.vals.shape[0] == self.__V.shape[0]:
            assert(False), "this should not happen, the fix is to remove a column and row of the laplacian"
            self.lap = self.lap[1:, 1:]

        return self.lap

    def compute_splu(self):
        print("i am computing splu")
        if self.cpu_splu is None:
            # st = time.time()
            s = scipy_splu(self.lap)
            # print(f"time to compute LU {time.time() - st}")
            # We are storing these attributes just in case we need to create a PoissonSolver on the GPU, they are useless for CPU case.
            self.cpu_splu = s
            self.__splu_L = s.L
            self.__splu_U = s.U
            self.__splu_perm_c = s.perm_c
            self.__splu_perm_r = s.perm_r
        return self.__splu_L,self.__splu_U,self.__splu_perm_c,self.__splu_perm_r

    def get_new_grad(self):
        grad = self.igl_grad.to_coo()
        self.igl_grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad.tocsc()),torch.float64)
        return self.igl_grad

def _convert_sparse_igl_grad_to_our_convention(input):
    '''
    The grad operator computed from igl.grad() results in a matrix of shape (3*#tri x #verts).
    It is packed such that all the x-coordinates are placed first, followed by y and z. As shown below

    ----------           ----------
    | x1 ...             | x1 ...
    | x2 ...             | y1 ...
    | x3 ...             | z1 ...
    | .                  | .
    | .                  | .
    | y1 ...             | x2 ...
    | y2 ...      ---->  | y2 ...
    | y3 ...             | z2 ...
    | .                  | .
    | .                  | .
    | z1 ...             | x3 ...
    | z2 ...             | y3 ...
    | z3 ...             | z3 ...
    | .                  | .
    | .                  | .
    ----------           ----------

    Note that this functionality cannot be computed trivially if because igl.grad() is a sparse tensor and as such
    slicing is not well defined for sparse matrices. the following code performs the above conversion and returns a
    torch.sparse tensor.
    Set check to True to verify the results by converting the matrices to dense and comparing it.
    '''
    assert type(input) == sp_csc, 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # horizontally stack row, col and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # sort along the row column

    # Separate out x, y and z blocks
    '''
    Note that for the grad operator there are exactly 3 non zero elements in a row
    '''
    L = T.shape[0]
    Tx = r_c_data[:L, :]
    Ty = r_c_data[L:2 * L, :]
    Tz = r_c_data[2 * L:3 * L, :]

    # align the y,z rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]
    Tz[:, 0] -= Tz[0, 0]

    # 'strech' the x,y,z rows so that they can be interleaved.
    Tx[:, 0] *= 3
    Ty[:, 0] *= 3
    Tz[:, 0] *= 3

    # interleave the y,z into x
    Ty[:, 0] += 1
    Tz[:, 0] += 2

    Tc = np.zeros((input.shape[0] * 3, 3))
    Tc[::3] = Tx
    Tc[1::3] = Ty
    Tc[2::3] = Tz

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])


class PoissonSolver:
    '''
    an object to compute gradients and solve poisson
    '''

    def __init__(self,grad,W,rhs,my_splu, lap=None, lap_pinned=None, components=None,
                 lap_pinned_rows = None, lap_pinned_cols=None, soft = False, my_splu_dense = None,
                 sparse = True):
        self.W = torch.from_numpy(W).double()
        self.grad = grad
        self.rhs = rhs
        self.my_splu = my_splu
        self.my_splu_dense = my_splu_dense
        self.lap = lap
        self.lap_pinned = lap_pinned
        self.lap_pinned_rows = lap_pinned_rows
        self.lap_pinned_cols = lap_pinned_cols
        self.components = components
        self.sparse_grad = grad
        self.sparse_rhs = rhs
        self.soft = soft
        self.sparse = sparse

    def to(self,device):
        self.W = self.W.to(device)
        self.sparse_grad = self.sparse_grad.to(device)
        self.sparse_rhs = self.sparse_rhs.to(device)
        if USE_CUPY or USE_CHOLESPY_GPU:
            self.lap = self.lap.to(device)
        return self

    def jacobians_from_vertices(self,V):
        res = _multiply_sparse_2d_by_dense_3d(self.sparse_grad, V).type_as(V)
        res = res.unsqueeze(2)
        return res.view(V.shape[0], -1, 3,3).transpose(2,3)

    def restrict_jacobians(self,D):
        assert isinstance(D, torch.Tensor) and len(D.shape) in [3, 4]
        assert D.shape[-1] == 3 and D.shape[-2] == 3
        assert isinstance(self.W, torch.Tensor) and len(self.W.shape) == 3
        assert self.W.shape[-1] == 2 and self.W.shape[-2] == 3

        if len(D.shape) == 4:
            DW = torch.einsum("abcd,bde->abce", (D, self.W.type_as(D)))
        else:
            DW = torch.einsum("abcd,bde->abce", (D.unsqueeze(0), self.W)).squeeze(0)

        if len(DW.shape)>4:
            DW = DW.squeeze(0)
        return DW

    def restricted_jacobians_from_vertices(self,V):
        return self.restrict_jacobians(self.jacobians_from_vertices(V))

    def solve_poisson(self,jacobians, updatedlap=False):
        # st = time.time()
        assert(len(jacobians.shape) == 4)
        assert(jacobians.shape[2] == 3 and jacobians.shape[3] == 3)

        # torch.cuda.synchronize()
        # st = time.time()

        # NOTE: This caches the splu object -- need to overwrite if there is weights
        if self.soft:
            if self.my_splu is None or self.my_splu_dense is None or updatedlap:
                if isinstance(self.lap,SparseMat):
                    self.my_splu = self.lap
                    self.my_splu_dense = torch.sparse_coo_tensor(self.lap.inds, self.lap.vals, (self.lap.n, self.lap.m)).to_dense()
                    # assert self.my_splu_dense.requires_grad, f"dense weighted laplacian should require grad"
        elif self.my_splu is None:
            self.my_splu = self.lap.to_cholesky()

        # NOTE: Jacobians go from (B x F x 3 x 3) => (B x F*3 x 3) with last two dimensions swapped (restriction dimension goes to last column)
        if self.soft:
            sol =  _predicted_jacobians_to_vertices_via_soft_poisson_solve(self.my_splu, self.my_splu_dense, self.sparse_rhs, jacobians.transpose(2, 3).reshape(jacobians.shape[0], -1, 3, 1).squeeze(3).contiguous(),
                                                                           sparse = self.sparse)
        else:
            sol = _predicted_jacobians_to_vertices_via_poisson_solve(self.my_splu, self.sparse_rhs, jacobians.transpose(2, 3).reshape(jacobians.shape[0], -1, 3, 1).squeeze(3).contiguous())

        if self.lap_pinned is not None:
            # NOTE: All component groups get pinned to origin
            for i in range(len(self.lap_pinned)):
                # The index should never be larger than length of array
                pidx = self.lap_pinned[i]
                assert sol.shape[1] >= pidx, f"Indexing error: trying to insert at {pidx} for array of size {sol.shape[1]}"
                sol = torch.cat([sol[:, :pidx], torch.zeros(sol.shape[0], 1, sol.shape[2]).type_as(sol), sol[:,pidx:]], dim=1)

            assert sol.shape[1] == len(self.components), f"Final UVs after undoing pinning {sol.shape[1]} should be same as components array {len(self.components)}"

            # pin_trans = torch.zeros(sol.shape, device=sol.device) # Should be same length
            # # Assign same component groups the same translation
            # for cpi in np.unique(self.components):
            #     cp_idx = np.where(self.components == cpi)[0]
            #     pin_trans[:,cp_idx] = torch.rand((sol.shape[0], 1, sol.shape[2]), device=sol.device)
            # sol += pin_trans

        # torch.cuda.synchronize()
        # print(f"POISSON LU + SOLVE FORWARD{time.time() - st}")
        c = torch.mean(sol, axis=1, keepdim=True)

        # print(f"time for poisson: {time.time() - st}" )
        return sol - c

    def pin_memory(self):
        return
        # self.W.pin_memory()
        # self.sparse_grad.pin_memory()
        # self.sparse_rhs.pin_memory()

# NOTE: Differential operators computed here!!!!
def poisson_system_matrices_from_mesh( V,F, dim=3,ttype = torch.float64, is_sparse=True, cpuonly=False, softpoisson=False):
    '''
    compute poisson matricees for a given mesh
    :param V vertices
    :param F faces
    :param dim: for now always 3 :)
    :param ttype the type of tensor (e.g., float,double)
    :param is_sparse: for now always true
    :return: a PoissonMatricese object holding the computed matrices
    '''

    assert type(dim) == int and dim in [2,3], f'Only two and three dimensional meshes are supported'
    assert type(is_sparse) == bool

    # Softpoisson mode: poisson matrix is triangle soup
    if softpoisson:
        fverts = V[F]
        vertices = fverts.reshape(-1, dim)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        soft = True
    else:
        vertices = V
        faces = F
        soft = False

    dim = 3
    is_sparse = is_sparse

    grad = igl.grad(vertices, faces)
    mass = _get_mass_matrix(vertices,faces,is_sparse)
    ## TODO 2D Case ##
    if dim == 2:
        grad = grad[:-grad.shape[0]//3,:]
        mass = mass[:-mass.shape[0]//3,:-mass.shape[0]//3]

    laplace = grad.T@mass@grad
    laplace = laplace.todense().astype(np.float64)

    np.fill_diagonal(laplace, 0)
    laplace[range(len(laplace)), range(len(laplace))] = -np.sum(laplace, axis=1).flatten()
    np.testing.assert_allclose(np.sum(laplace, axis=1), 0, atol=1e-4)

    # For soft poisson, we assume the surface is connected
    lap_pinned_rows = None
    lap_pinned_cols = None
    if softpoisson:
        lap_pinned = np.array([0])

        # NOTE: We save rows and cols of deleted lap so we can update correctly later (order is add row then col)
        lap_pinned_rows = [laplace[0, 1:]]
        lap_pinned_cols = [laplace[:, 0]]
        components = np.zeros(vertices.shape[0])
        laplace = np.delete(np.delete(laplace, lap_pinned, axis=0), lap_pinned, axis=1) # Remove row and column
    else:
        component_i, components, component_sizes = igl.connected_components(igl.adjacency_matrix(faces))
        # Find first index for each component
        lap_pinned = []
        for i in range(component_sizes.size):
            cp_idx = np.where(components == i)[0][0]
            lap_pinned.append(cp_idx)
        lap_pinned = np.array(lap_pinned)
        laplace = np.delete(np.delete(laplace, lap_pinned, axis=0), lap_pinned, axis=1) # Remove row and column

    # Convert back to sparse matrix
    laplace = sparse.csr_matrix(laplace)

    rhs = grad.T@mass
    b1,b2,_ = igl.local_basis(V,F)
    # NOTE: w defines the restriction operator to the local tangent basis per triangle
    w = np.stack((b1,b2),axis=-1)
    # print(time.time() - s)

    # Remove pinned indices from rhs as well
    rhs = rhs.todense()
    rhs = np.delete(rhs, lap_pinned, axis=0)
    rhs = sparse.csr_matrix(rhs)

    rhs = rhs.tocoo()
    grad = grad.tocsc()
    laplace = laplace.tocoo()

    # if is_sparse:
    #     laplace = laplace.tocoo()
    # else:
    #     laplace = laplace.toarray()
    #     rhs = rhs.toarray()
    #     grad = grad.toarray()

    # Undoing the pinning just means inserting back the values in index order
    lap_pinned = np.sort(lap_pinned)

    grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad), torch.float64)
    poissonbuilder =  PoissonSystemMatrices(V=V,F=F,grad=grad,
              rhs=SparseMat.from_coo(rhs, torch.float64), w=w,
              ttype=ttype,is_sparse=is_sparse, soft=soft,
              lap=SparseMat.from_coo(laplace, torch.float64),
              lap_pinned_rows = lap_pinned_rows,
              lap_pinned_cols = lap_pinned_cols,
              cpuonly=cpuonly, lap_pinned=lap_pinned, components=components,
              sparse = is_sparse)
    # poissonbuilder.get_new_grad()
    return poissonbuilder

def _get_mass_matrix(vertices,faces,is_sparse):

    d_area = igl.doublearea(vertices,faces)
    d_area = np.hstack((d_area, d_area, d_area))
    if is_sparse:
        return sp_csc(diags(d_area))
    return diags(d_area)




class SPLUSolveLayer(torch.autograd.Function):
    '''
    Implements the SPLU solve as a differentiable layer, with a forward and backward function
    '''

    @staticmethod
    def forward(ctx, solver, b):
        '''
        override forward function
        :param ctx: context object (to keep the lu object for the backward pass)
        :param lu: splu object
        :param b: right hand side, could be a vector or matrix
        :return: the vector or matrix x which holds lu.solve(b) = x
        '''
        assert isinstance(b, torch.Tensor)
        assert b.shape[-1] >= 1 and b.shape[-1] <= 3, f'got shape {b.shape} expected last dim to be in range 1-3'
        b = b.contiguous()
        ctx.solver = solver

        # st = time.time()
        vertices = SPLUSolveLayer.solve(solver, b).type_as(b)
        # print(f"FORWARD SOLVE {time.time() - st}")

        assert not torch.isnan(vertices).any(), "Nan in the forward pass of the POISSON SOLVE"
        return vertices

    def backward(ctx, grad_output):
        '''
        overrides backward function
        :param grad_output: the gradient to be back-propped
        :return: the outgoing gradient to be back-propped
        '''

        assert isinstance(grad_output, torch.Tensor)
        assert grad_output.shape[-1] >= 1 and grad_output.shape[
            -1] <= 3, f'got shape {grad_output.shape} expected last dim to be in range 1-3'
        # when backpropping, if a layer is linear with matrix M, x ---> Mx, then the backprop of gradient g is M^Tg
        # in our case M = A^{-1}, so the backprop is to solve x = A^-T g.
        # Because A is symmetric we simply solve A^{-1}g without transposing, but this will break if A is not symmetric.
        # st = time.time()
        grad_output = grad_output.contiguous()
        grad = SPLUSolveLayer.solve(ctx.solver,
                                          grad_output)
        # print(f"BACKWARD SOLVE {time.time() - st}")
        # At this point we perform a NAN check because the backsolve sometimes returns NaNs.
        assert not torch.isnan(grad).any(),  "Nan in the backward pass of the POISSON SOLVE"

        if USE_CUPY:
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            del ctx.lu

        return None, grad

    @staticmethod
    def solve(solver, b):
        '''
        solve the linear system defined by an SPLU object for a given right hand side. if the RHS is a matrix, solution will also be a matrix.
        :param solver: the splu object (LU decomposition) or cholesky object
        :param b: the right hand side to solve for
        :return: solution x which satisfies Ax = b where A is the poisson system lu describes
        '''

        if  USE_CUPY:
            b_cupy = cupy.fromDlpack(to_dlpack(b))
            with cupy.cuda.Device(solver.device()):
                # this will hold the solution
                sol = cupy.ndarray(b_cupy.shape)
                for i in range(b_cupy.shape[2]):  # b may have multiple columns, solve for each one
                    b2d = b_cupy[..., i]  # cupy.expand_dims(b_cpu[...,i],2)
                    s = solver.solve(b2d.T).T
                    sol[:, :, i] = s
            # # # convert back to torch
            res = from_dlpack(sol.toDlpack())
            # np.save("res_gpu.npy", res.cpu().numpy())
            # res = torch.zeros((1, 6889, 3), device=b.device)+  torch.mean(b)

            return res.type_as(b.type())

        elif USE_SCIPY:
            #only CPU
            # st = time.time()
            assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            sol = solver.solve(b[0].double().cpu().numpy())
            res = torch.from_numpy(sol).to(b.device).reshape(b.shape)
            # print(time.time() - st)
            return res.type_as(b).contiguous()

            # Legacy code, I don't understand what is the reason for having a for loop
            # sol = np.ndarray(b.shape)
            # for i in range(b.shape[2]):  # b may have multiple columns, solve for each one
            #     b2d = b[..., i]  # cupy.expand_dims(b_cpu[...,i],2)
            #     s = lu.solve(b2d.double().cpu().float().numpy().T).T
            #     sol[:, :, i] = s
            # res = torch.from_numpy(sol).to(b.device)
            # # np.save("res_cpu.npy", sol)
            # print(f"time {time.time() - st}" )
        elif USE_CHOLESPY_GPU:
            # torch.cuda.synchronize()
            # # st = time.time()
            # assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            # b = b.squeeze().double()
            # x = torch.zeros_like(b)
            # solver.solve(b, x)
            # # torch.cuda.synchronize()
            # # print(f"time cholescky GPU {time.time() - st}" )
            # return x.contiguous().unsqueeze(0)
            # st = time.time()
            # print(b.get_device(), b.shape)
            b = b.double().contiguous()
            c = b.permute(1,2,0).contiguous()
            c = c.view(c.shape[0], -1)
            x = torch.zeros_like(c)
            solver.solve(c, x)
            x = x.view(b.shape[1], b.shape[2], b.shape[0])
            x = x.permute(2,0,1).contiguous()
            # torch.cuda.synchronize()
            # print(f"time cholescky GPU {time.time() - st}" )
            return x.contiguous()

        elif USE_CHOLESPY_CPU:
            # st = time.time()
            assert(b.shape[0]==1), "Need to code parrallel implem on the first dim"
            b = b.squeeze()
            b_cpu = b.cpu()
            x = torch.zeros_like(b_cpu)
            solver.solve(b_cpu, x)
            # print(f"time cholescky CPU {time.time() - st}" )
            return x.contiguous().to(b.device).unsqueeze(0)


        return res.type_as(b)

class SoftPoissonSolveLayer(torch.autograd.Function):
    '''
    Implements the Soft Poisson solve as a differentiable layer, with a forward and backward function
    '''

    @staticmethod
    def forward(ctx, sparsemat, densemat, b):
        '''
        override forward function
        :param ctx: context object (to keep densemat for the backward pass)
        :param sparsemat: sparse wL matrix for passing into cholespy
        :param densemat: dense wL matrix for passing gradient through in backwards pass
        :param b: right hand side, could be a vector or matrix
        :return: the vector or matrix x which holds lu.solve(b) = x
        '''
        assert isinstance(b, torch.Tensor)
        assert b.shape[-1] >= 1 and b.shape[-1] <= 3, f'got shape {b.shape} expected last dim to be in range 1-3'

        solver = CholeskySolverD(sparsemat.n, sparsemat.inds[0,:], sparsemat.inds[1,:], sparsemat.vals, MatrixType.COO)
        ctx.solver = solver

        ctx.save_for_backward(densemat.detach(), b.detach())

        # st = time.time()
        vertices = SoftPoissonSolveLayer.solve(solver, b).type_as(b)
        # print(f"FORWARD SOLVE {time.time() - st}")

        assert not torch.isnan(vertices).any(), "Nan in the forward pass of the POISSON SOLVE"
        return vertices

    def backward(ctx, grad_output):
        '''
        overrides backward function
        :param grad_output: the gradient to be back-propped
        :return: the outgoing gradient to be back-propped
        '''

        assert isinstance(grad_output, torch.Tensor)

        # when backpropping, if a layer is linear with matrix M, x ---> Mx, then the backprop of gradient g is M^Tg
        # in our case M = A^{-1}, so the backprop is to solve x = A^-T g.
        # Because A is symmetric we simply solve A^{-1}g without transposing, but this will break if A is not symmetric.
        # st = time.time()
        densewL, b = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_b = SoftPoissonSolveLayer.solve(ctx.solver,
                                          grad_output)

        grad_wl = grad_output @ b.transpose(-2, -1)

        # (A^-1 G^T)^T = G A^-1 (A is symmetric)
        grad_wl = -SoftPoissonSolveLayer.solve(ctx.solver, grad_wl.transpose(-2, -1)).transpose(-2, -1)
        grad_wl = SoftPoissonSolveLayer.solve(ctx.solver, grad_wl)

        # Grad of inverse is -A^-2 @ dL/dA^-1
        # NOTE: A^-1 can only be calculated with n <= 128, so need general solution here instead
        # grad_wl = grad_output @ b.transpose(0,1)
        # ident = torch.eye(len(densewL)).to(densewL.device).double().contiguous()
        # wL_inv = torch.zeros_like(densewL).contiguous()
        # solver = ctx.solver
        # solver.solve(ident, wL_inv)
        # grad_wl = -wL_inv.transpose(0,1) @ grad_wl @ wL_inv.transpose(0,1)

        # Debugging gradients
        # print(f"Grad output quintile: {torch.quantile(grad_output, torch.linspace(0,1,5).to(grad_output.device).double())}")
        # print(f"grad_wl quintile: {torch.quantile(grad_wl, torch.linspace(0,1,5).to(grad_wl.device).double())}")
        # print(f"grad_b quintile: {torch.quantile(grad_b, torch.linspace(0,1,5).to(grad_b.device).double())}")

        # At this point we perform a NAN check because the backsolve sometimes returns NaNs.
        assert not torch.isnan(grad_b).any(),  "Nan in the backward pass of the POISSON SOLVE"
        assert not torch.isnan(grad_wl).any(),  "Nan in the backward pass of the POISSON SOLVE"

        return None, grad_wl, grad_b

    @staticmethod
    def solve(solver, b):
        '''
        solve the linear system defined by an SPLU object for a given right hand side. if the RHS is a matrix, solution will also be a matrix.
        :param solver: the splu object (LU decomposition) or cholesky object
        :param b: the right hand side to solve for
        :return: solution x which satisfies Ax = b where A is the poisson system lu describes
        '''

        b = b.double().contiguous()
        c = b.squeeze()
        c = b.permute(1,2,0).contiguous()
        c = c.view(c.shape[0], -1) # NOTE: Collapses the batch dimension (assumes same laplacian for multiple preds of Jacobs)
        x = torch.zeros_like(c)
        solver.solve(c, x)
        x = x.view(b.shape[1], b.shape[2], b.shape[0])
        x = x.permute(2,0,1).contiguous()

        return x.contiguous()

def _predicted_jacobians_to_vertices_via_soft_poisson_solve(softlap_sparse, softlap_dense, rhs, jacobians, sparse=True):
    '''
    convert the predictions to the correct convention and feed it to the poisson solve
    '''

    def _batch_rearrange_input(input):
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            # Batched input
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3]
            P[:, k:2 * k, :] = input[:, 1::3]
            P[:, 2 * k:, :] = input[:, 2::3]

        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k:2 * k, :] = input[1::3]
            P[2 * k:, :] = input[2::3]

        return P

    def _list_rearrange_input(input):
        assert isinstance(input, list) and all([isinstance(x, torch.Tensor) and len(x.shape) in [2, 3] for x in input])
        P = []
        for p in input:
            P.append(_batch_rearrange_input(p))
        return P

    if isinstance(jacobians, list):
        P = _list_rearrange_input(jacobians)
    else:
        P = _batch_rearrange_input(jacobians)

    # return solve_poisson(Lap, rhs, P)
    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3

    # torch.cuda.synchronize()
    # st = time.time()
    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)
    input_to_solve = input_to_solve.contiguous()

    # NOTE: Use different solve layer that takes as input the sparsemat values directly and constructs
    #       the Lap splu solver from there => need to pass gradients back using product rule to both
    #       the soft Laplacian AND the jacobians
    if not sparse:
        out = torch.linalg.solve(softlap_dense, input_to_solve) # B x V x 2
    else:
        out = SoftPoissonSolveLayer.apply(softlap_sparse, softlap_dense, input_to_solve) # B x V x 2

    return out.type_as(jacobians)

def _predicted_jacobians_to_vertices_via_poisson_solve(Lap, rhs, jacobians):
    '''
    convert the predictions to the correct convention and feed it to the poisson solve
    '''

    def _batch_rearrange_input(input):
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            # Batched input
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3]
            P[:, k:2 * k, :] = input[:, 1::3]
            P[:, 2 * k:, :] = input[:, 2::3]

        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k:2 * k, :] = input[1::3]
            P[2 * k:, :] = input[2::3]

        return P

    def _list_rearrange_input(input):
        assert isinstance(input, list) and all([isinstance(x, torch.Tensor) and len(x.shape) in [2, 3] for x in input])
        P = []
        for p in input:
            P.append(_batch_rearrange_input(p))
        return P

    if isinstance(jacobians, list):
        P = _list_rearrange_input(jacobians)
    else:
        P = _batch_rearrange_input(jacobians)

    # return solve_poisson(Lap, rhs, P)
    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3

    # torch.cuda.synchronize()
    # st = time.time()
    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)

    # NOTE: Use different solve layer that takes as input the sparsemat values directly and constructs
    #       the Lap splu solver from there => need to pass gradients back using product rule to both
    #       the soft Laplacian AND the jacobians
    out = SPLUSolveLayer.apply(Lap, input_to_solve) # B x V x 2

    return out.type_as(jacobians)

def _multiply_sparse_2d_by_dense_3d(mat, B):
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret







class MyCuSPLU:
    '''
    implmentation of SPLU on the gpu via CuPy
    '''
    def __init__(self, L, U, perm_c=None, perm_r=None):
        # with cupy.cuda.Device(device):
        self.__orgL = L
        self.__orgU = U
        # self.L = csr_matrix(L)
        # self.U = csr_matrix(U)
        self.L = None
        self.U = None
        self.perm_c = perm_c
        self.perm_r = perm_r
        # self.splu = cu_splu(csr_matrix(lap))
        # self.L = self.splu.L
        # self.U = self.splu.U
        # self.perm_c = self.splu.perm_c
        # self.perm_r = self.splu.perm_r
        self.__device = None

    def to(self, device):
        # assumes to receive a pytorch device object that has a "index" field
        # print(device)
        # if(self.__device is None):
        #     raise Exception()
        self.__device = device.index
        with cupy.cuda.Device(self.__device):
            # self.__orgL = cupy.asarray(self.__orgL)
            # self.__orgU = cupy.asarray(self.__orgU)
            self.L = csr_matrix(self.__orgL)
            self.U = csr_matrix(self.__orgU)
        return self

    def device(self):
        return self.__device

    def solve(self, b):
        """ an attempt to use SuperLU data to efficiently solve
            Ax = Pr.T L U Pc.T x = b
             - note that L from SuperLU is in CSC format solving for c
               results in an efficiency warning
            Pr . A . Pc = L . U
            Lc = b      - forward solve for c
             c = Ux     - then back solve for x
        """

        assert self.__device is not None, "need to explicitly call to() before solving"
        if USE_UGLY_PATCH_FOR_CUPY_ERROR:
            with cupy.cuda.Device(0):
                b[:1, :1].copy()[:, :1]

        with cupy.cuda.Device(self.__device):
            b = cupy.array(b)
            if self.perm_r is not None:
                b_old = b.copy()
                b[self.perm_r] = b_old

        assert b.device.id == self.__device, "got device" + str(b.device.id) + "instead of" + str(self.__device)
        # st = time.time()
        try:  # unit_diagonal is a new kw
            c = spsolve_triangular(self.L, b, lower=True, unit_diagonal=True, overwrite_b=True)
        except TypeError:
            c = spsolve_triangular(self.L, b, lower=True, overwrite_b=True)
        px = spsolve_triangular(self.U, c, lower=False, overwrite_b=True)
        # print(f"time for spsolve_triangular GPU: {time.time() - st}" )

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        # print(f'used: {mempool.used_bytes()}')
        # print(f'total: {mempool.total_bytes()}')
        return px


class MyCuSPLU_CPU:
    '''
    implmentation of SPLU on the gpu via CuPy
    '''
    def __init__(self, L, U, perm_c=None, perm_r=None):
        # with cupy.cuda.Device(device):
        self.__orgL = L
        self.__orgU = U
        # self.L = csr_matrix(L)
        # self.U = csr_matrix(U)
        self.L = L
        self.U = U
        # self.L = L.tocsr()
        # self.U = U.tocsr()
        self.perm_c = perm_c
        self.perm_r = perm_r
        # self.splu = cu_splu(csr_matrix(lap))
        # self.L = self.splu.L
        # self.U = self.splu.U
        # self.perm_c = self.splu.perm_c
        # self.perm_r = self.splu.perm_r
        self.__device = 'cpu'

    def to(self, device):
        # assumes to receive a pytorch device object that has a "index" field
        # print(device)
        # if(self.__device is None):
        #     raise Exception()
        # self.__device = device.index
        # with cupy.cuda.Device(self.__device):
        #     # self.__orgL = cupy.asarray(self.__orgL)
        #     # self.__orgU = cupy.asarray(self.__orgU)
        #     self.L = csr_matrix(self.__orgL)
        #     self.U = csr_matrix(self.__orgU)
        return self

    def device(self):
        return self.__device

    def solve(self, b):
        """ an attempt to use SuperLU data to efficiently solve
            Ax = Pr.T L U Pc.T x = b
             - note that L from SuperLU is in CSC format solving for c
               results in an efficiency warning
            Pr . A . Pc = L . U
            Lc = b      - forward solve for c
             c = Ux     - then back solve for x
        """


        # Could be done on GPU
        if self.perm_r is not None:
            b_old = b.copy()
            b[self.perm_r] = b_old
        # ,  permc_spec="NATURAL"
        # ,  permc_spec="NATURAL"
        # ,  permc_spec="NATURAL"
        st = time.time()
        # try:  # unit_diagonal is a new kw
        #     c = spsolve_triangular(self.L, b, lower=True, unit_diagonal=True, overwrite_b=True)
        # except TypeError:
        #     c = spsolve_triangular(self.L, b, lower=True, overwrite_b=True)
        # px = spsolve_triangular(self.U, c, lower=False, overwrite_b=True)
        try:  # unit_diagonal is a new kw
            c = spsolve(self.L, b, permc_spec="NATURAL")
        except TypeError:
            c = spsolve(self.L, b, permc_spec="NATURAL")
        px = spsolve(self.U, c, permc_spec="NATURAL")
        # # (self.L *  c) - b / np.norm(b)
        print(f"time for spsolve_triangular CPU: {time.time() - st}" )

        if self.perm_c is None:
            return px
        px = px[self.perm_c]

        # print(f'used: {mempool.used_bytes()}')
        # print(f'total: {mempool.total_bytes()}')
        return px

        # return cupy.asnumpy(px)
