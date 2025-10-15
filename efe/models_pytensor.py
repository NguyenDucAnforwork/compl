"""
Define all model classes following the definition of Abstract_Model.
Updated to use PyTensor instead of Theano.
"""
import pytensor
import pytensor.tensor as pt
import numpy as np
from pytensor.compile import SharedVariable

data_type = 'float32'
pytensor.config.floatX = data_type

# Helper function to replace randn
def bce_with_logits(logits, targets):
    # targets ∈ {0,1}, dtype float32
    targets = pt.cast(targets, logits.dtype)
    # Công thức ổn định số: max(x,0) - x*y + log1p(exp(-|x|))
    return (pt.maximum(logits, 0) - logits * targets + pt.log1p(pt.exp(-pt.abs(logits)))).mean()

def randn(*shape):
    return np.random.randn(*shape).astype(data_type)

def L2_proj(x):
    """L2 normalization"""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Abstract_Model(object):

    def __init__(self):
        self.name = self.__class__.__name__
        
        self.pred_func = None
        self.pred_func_compiled = None
        self.loss_func = None
        self.regul_func = None
        self.loss_to_opt = None
        
        # Symbolic variables for training values
        self.ys = pt.vector('ys')
        self.rows = pt.lvector('rows')
        self.cols = pt.lvector('cols')
        self.tubes = pt.lvector('tubes')

        # Dimensions
        self.n = 0  # Number of subject entities
        self.m = 0  # Number of relations
        self.l = 0  # Number of object entities
        self.k = 0  # Rank
        self.nb_params = 0

    def allocate_params(self):
        nb_params = 0
        params = self.get_init_params()
        for name, val in params.items():
            setattr(self, name, pytensor.shared(val, name=name))
            nb_params += val.size
        self.nb_params = nb_params

    def get_pred_symb_vars(self):
        return [self.rows, self.cols, self.tubes]

    def get_pred_args(self, test_idxs):
        return [test_idxs[:, 0], test_idxs[:, 1], test_idxs[:, 2]]

    def compile_prediction(self):
        """Compile the prediction function"""
        self.pred_func_compiled = pytensor.function(
            self.get_pred_symb_vars(), 
            self.pred_func
        )

    def predict(self, test_idxs):
        """Predict on test indices"""
        return self.pred_func_compiled(*self.get_pred_args(test_idxs))

    def get_init_params(self):
        """Abstract method - must be implemented by child classes"""
        raise NotImplementedError

    def define_loss(self):
        """Abstract method - must be implemented by child classes"""
        raise NotImplementedError

    def set_dims(self, train_triples, hparams):
        """Set dimensions from training data"""
        self.n = max(train_triples.indexes[:,0]) + 1
        self.m = max(train_triples.indexes[:,1]) + 1  
        self.l = max(train_triples.indexes[:,2]) + 1
        self.k = hparams.embedding_size

    def fit(self, train_triples, valid_triples, hparams, n=0, m=0, l=0, scorer=None):
        """Simple fit method to maintain compatibility"""
        print(f"Training {self.name} with PyTensor...")
        
        # Set dimensions
        if n == 0:
            self.set_dims(train_triples, hparams)
        else:
            self.n, self.m, self.l, self.k = n, m, l, hparams.embedding_size
        
        # Initialize model
        self.allocate_params()
        self.define_loss()
        # self.compile_prediction()
        
        print(f"Model initialized with {self.nb_params} parameters")
        print(f"Dimensions: n={self.n}, m={self.m}, l={self.l}, k={self.k}")
        
        # Simple training loop
        learning_rate = min(hparams.learning_rate, 0.01)
        lmbda = hparams.lmbda
        max_iter = min(hparams.max_iter, 2500)  # Limit iterations for quick test
        batch_size = min(hparams.batch_size, 1000)  # Limit batch size
        
        # Create simple training function
        loss_total = self.loss + lmbda * self.regul_func
        
        # Get parameters
        params = []
        if hasattr(self, 'e'):
            params.extend([self.e, self.r])
        if hasattr(self, 'e1'):
            params.extend([self.e1, self.e2, self.r1, self.r2])
        
        grads = pytensor.grad(loss_total, params)
        # 👇 Clip để tránh nổ gradient
        grads = [pt.clip(g, -5.0, 5.0) for g in grads]
        updates = [(p, p - learning_rate * g) for p, g in zip(params, grads)]
        
        train_fn = pytensor.function(
            [self.rows, self.cols, self.tubes, self.ys],
            [loss_total, self.loss],
            updates=updates
        )

        # print params initial values
        print("***INITIAL VALUES: ***", self.e.get_value()[0])
        
        # Simple training with positive samples only
        n_samples = min(len(train_triples.indexes), 1000)  # Limit samples
        
        for epoch in range(max_iter):
            pos_idx = np.random.choice(len(train_triples.indexes),
                                    min(batch_size, n_samples), replace=False)
            pr = train_triples.indexes[pos_idx, 0].astype('int64')  # s
            pc = train_triples.indexes[pos_idx, 1].astype('int64')  # r
            ptb = train_triples.indexes[pos_idx, 2].astype('int64') # o

            # --- Negative sampling: corrupt object
            n_entities = max(self.n, self.l)  # DistMult/ComplEx dùng e size = max(n,l)
            neg_ratio = max(1, getattr(hparams, "neg_ratio", 1))
            nr = np.repeat(pc, neg_ratio)
            ns = np.repeat(pr, neg_ratio)
            # sample random objects
            no = np.random.randint(0, n_entities, size=len(ns)).astype('int64')  # 👈 dùng n_entities

            # (tùy chọn) tránh false negative đơn giản
            # for t in range(len(ns)):
            #     while (ns[t], nr[t]) in scorer.known_obj_triples and \
            #           no[t] in scorer.known_obj_triples[(ns[t], nr[t])]:
            #         no[t] = np.random.randint(0, self.n, dtype='int64')

            # --- Gộp positives + negatives
            rows = np.concatenate([pr, ns])
            cols = np.concatenate([pc, nr])
            tubes = np.concatenate([ptb, no])
            ys = np.concatenate([np.ones(len(pr), dtype='float32'),
                                np.zeros(len(ns), dtype='float32')])

            batch_ys = ys.astype('float32')  # đảm bảo float32
            total_loss, main_loss = train_fn(rows, cols, tubes, ys)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")



class DistMult_Model(Abstract_Model):
    """DistMult model"""

    def __init__(self):
        super(DistMult_Model, self).__init__()
        self.name = self.__class__.__name__
        self.e = None
        self.r = None

    def get_init_params(self):
        params = {
            'e': randn(max(self.n, self.l), self.k),
            'r': randn(self.m, self.k)
        }
        return params

    def define_loss(self):
        logits = pt.sum(self.e[self.rows, :] * self.r[self.cols, :] * self.e[self.tubes, :], 1)
        self.pred_func = logits  # giữ logits để ranking dùng score thô
        self.loss = bce_with_logits(logits, self.ys)
        self.regul_func = (
            pt.sqr(self.e[self.rows, :]).mean() +
            pt.sqr(self.r[self.cols, :]).mean() +
            pt.sqr(self.e[self.tubes, :]).mean()
        )




class Complex_Model(Abstract_Model):
    """Factorization in complex numbers"""

    def __init__(self):
        super(Complex_Model, self).__init__()
        self.name = self.__class__.__name__
        self.e1 = None
        self.e2 = None
        self.r1 = None
        self.r2 = None

    def get_init_params(self):
        params = {
            'e1': randn(max(self.n, self.l), self.k),
            'e2': randn(max(self.n, self.l), self.k),
            'r1': randn(self.m, self.k),
            'r2': randn(self.m, self.k)
        }
        return params

    def define_loss(self):
        self.pred_func = (
            pt.sum(self.e1[self.rows, :] * self.r1[self.cols, :] * self.e1[self.tubes, :], 1) +
            pt.sum(self.e2[self.rows, :] * self.r1[self.cols, :] * self.e2[self.tubes, :], 1) +
            pt.sum(self.e1[self.rows, :] * self.r2[self.cols, :] * self.e2[self.tubes, :], 1) -
            pt.sum(self.e2[self.rows, :] * self.r2[self.cols, :] * self.e1[self.tubes, :], 1)
        )
        
        self.loss = pt.sqr(self.ys - self.pred_func).mean()
        
        self.regul_func = (
            pt.sqr(self.e1[self.rows, :]).mean() +
            pt.sqr(self.e2[self.rows, :]).mean() +
            pt.sqr(self.e1[self.tubes, :]).mean() +
            pt.sqr(self.e2[self.tubes, :]).mean() +
            pt.sqr(self.r1[self.cols, :]).mean() +
            pt.sqr(self.r2[self.cols, :]).mean()
        )