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
    # đảm bảo dtype int64 (khớp với lvector)
        if test_idxs.dtype != np.int64:
            test_idxs = test_idxs.astype(np.int64)

        # nếu chưa compile thì compile ngay
        if self.pred_func_compiled is None:
            self.compile_prediction()

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
        self.compile_prediction()
        
        print(f"Model initialized with {self.nb_params} parameters")
        print(f"Dimensions: n={self.n}, m={self.m}, l={self.l}, k={self.k}")
        
        # Hyperparameters
        learning_rate = hparams.learning_rate
        lmbda = hparams.lmbda
        max_iter = hparams.max_iter
        batch_size = hparams.batch_size
        neg_ratio = getattr(hparams, 'neg_ratio', 1)

        # Get trainable parameters
        params = []
        if hasattr(self, 'e'):
            params.extend([self.e, self.r])
        if hasattr(self, 'e1'):
            params.extend([self.e1, self.e2, self.r1, self.r2])

        # --- Adagrad Optimizer Setup ---
        # Create shared variables for Adagrad gradient accumulators
        accumulators = [
            pytensor.shared(np.zeros(p.get_value(borrow=True).shape, dtype=data_type))
            for p in params
        ]
        
        # Define loss and gradients
        loss_total = self.loss + lmbda * self.regul_func
        grads = pytensor.grad(loss_total, params)
        
        # Adagrad update rule
        updates = []
        for p, g, a in zip(params, grads, accumulators):
            # Clip gradient to prevent explosion
            clipped_g = pt.clip(g, -5.0, 5.0)
            # Update accumulator
            new_a = a + clipped_g ** 2
            updates.append((a, new_a))
            # Update parameter
            updates.append((p, p - (learning_rate / (pt.sqrt(new_a) + 1e-8)) * clipped_g))
        
        # Compile the training function
        train_fn = pytensor.function(
            [self.rows, self.cols, self.tubes, self.ys],
            [loss_total, self.loss],
            updates=updates
        )

        print(f"***INITIAL VALUES (sample): *** {self.e.get_value()[0, :5]}")
        
        # Simple training with positive samples only
        n_train_samples = len(train_triples.indexes)  # Limit samples
        
        for epoch in range(max_iter):
            # --- Data Sampling (FIXED) ---
            # Sample a batch from the ENTIRE training set
            pos_indices = np.random.choice(n_train_samples, batch_size, replace=False)
            pos_batch = train_triples.indexes[pos_indices]

            # --- Negative Sampling ---
            # Corrupt either subject or object
            corrupted_batch = pos_batch.copy()
            n_entities = self.e.get_value(borrow=True).shape[0]
            
            # For each positive sample, create 'neg_ratio' negative samples
            num_negs = batch_size * neg_ratio
            random_entities = np.random.randint(0, n_entities, size=num_negs)
            
            # Repeat positive triples to match number of negative samples
            repeated_pos = np.repeat(pos_batch, neg_ratio, axis=0)
            
            # Decide whether to corrupt subject (0) or object (1)
            sub_or_obj_corr = np.random.randint(2, size=num_negs)
            
            mask_sub = (sub_or_obj_corr == 0)
            repeated_pos[mask_sub, 0] = random_entities[mask_sub]
            
            mask_obj = (sub_or_obj_corr == 1)
            repeated_pos[mask_obj, 2] = random_entities[mask_obj]
            
            neg_batch = repeated_pos

            # --- Prepare batch for training ---
            full_batch_rows = np.concatenate([pos_batch[:, 0], neg_batch[:, 0]]).astype('int64')
            full_batch_cols = np.concatenate([pos_batch[:, 1], neg_batch[:, 1]]).astype('int64')
            full_batch_tubes = np.concatenate([pos_batch[:, 2], neg_batch[:, 2]]).astype('int64')
            
            ys = np.concatenate([
                np.ones(batch_size, dtype=data_type),
                np.zeros(num_negs, dtype=data_type)
            ])
            
            total_loss, main_loss = train_fn(full_batch_rows, full_batch_cols, full_batch_tubes, ys)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f} (Main Loss: {main_loss:.4f})")



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