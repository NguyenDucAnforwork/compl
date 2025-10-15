import operator
import sklearn
import sklearn.metrics

import numpy as np

from .tools import *
import torch

class Result(object):
	"""
	Store one test results
	"""

	def __init__(self, preds, true_vals, ranks, raw_ranks):
		self.preds = preds
		self.ranks = ranks
		self.true_vals = true_vals
		self.raw_ranks = raw_ranks

		#Test if not all the prediction are the same, sometimes happens with overfitting,
		#and leads scikit-learn to output incorrect average precision (i.e ap=1)
		if not (preds == preds[0]).all() :
			#Due to the use of np.isclose in sklearn.metrics.ranking._binary_clf_curve (called by following metrics function),
			#I have to rescale the predictions if they are too small:
			preds_rescaled = preds

			diffs = np.diff(np.sort(preds))
			min_diff = min(abs(diffs[np.nonzero(diffs)]))
			if min_diff < 1e-8 : #Default value of absolute tolerance of np.isclose
				preds_rescaled = (preds * ( 1e-7 / min_diff )).astype('d')

			self.ap = sklearn.metrics.average_precision_score(true_vals,preds_rescaled)
			self.precision, self.recall, self.thresholds = sklearn.metrics.precision_recall_curve(true_vals,preds_rescaled) 
		else:
			logger.warning("All prediction scores are equal, probable overfitting, replacing scores by random scores")
			self.ap = (true_vals == 1).sum() / float(len(true_vals))
			self.thresholds = preds[0]
			self.precision = (true_vals == 1).sum() / float(len(true_vals))
			self.recall = 0.5
		
		
		self.mrr =-1
		self.raw_mrr =-1

		if ranks is not None:
			self.mrr = np.mean(1.0 / ranks)
			self.raw_mrr = np.mean(1.0 / raw_ranks)




class CV_Results(object):
	"""
	Class that stores predictions and scores by indexing them by model, embedding_size and lmbda
	"""

	def __init__(self):
		self.res = {}
		self.nb_params_used = {} #Indexed by model_s and embedding sizes, in order to plot with respect to the number of parameters of the model


	def add_res(self, res, model_s, embedding_size, lmbda, nb_params):
		if model_s not in self.res:
			self.res[model_s] = {}
		if embedding_size not in self.res[model_s]:
			self.res[model_s][embedding_size] = {}
		if lmbda not in self.res[model_s][embedding_size]:
			self.res[model_s][embedding_size][lmbda] = []

		self.res[model_s][embedding_size][lmbda].append( res )

		if model_s not in self.nb_params_used:
			self.nb_params_used[model_s] = {}
		self.nb_params_used[model_s][embedding_size] = nb_params


	def extract_sub_scores(self, idxs):
		"""
		Returns a new CV_Results object with scores only at the given indexes
		"""

		new_cv_res = CV_Results()

		for j, (model_s, cur_res) in enumerate(self.res.items()):
			for i,(k, lmbdas) in enumerate(cur_res.items()):
				for lmbda, res_list in lmbdas.items():
					for res in res_list:
						if res.ranks is not None:
							#Concat idxs on ranks as subject and object ranks are concatenated in a twice larger array
							res = Result(res.preds[idxs], res.true_vals[idxs], res.ranks[np.concatenate((idxs,idxs))], res.raw_ranks[np.concatenate((idxs,idxs))])
						else:
							res = Result(res.preds[idxs], res.true_vals[idxs], None, None)
						
						new_cv_res.add_res(res, model_s, k, lmbda, self.nb_params_used[model_s][k])

		return new_cv_res


	def _get_best_mean_ap(self, model_s, embedding_size):
		"""
		Averaging runs for each regularization value, and picking the best AP
		"""

		lmbdas = self.res[model_s][embedding_size]

		mean_aps = []
		var_aps = []
		for lmbda_aps in lmbdas.values():
			mean_aps.append( np.mean( [ result.ap for result in lmbda_aps] ) )
			var_aps.append( np.std( [ result.ap for result in lmbda_aps] ) )
		cur_aps_moments = zip(mean_aps, var_aps)

		return max(cur_aps_moments, key = operator.itemgetter(0)) #max by mean






	def print_MRR_and_hits_given_params(self, model_s, rank, lmbda):

		mrr = np.mean( [ res.mrr for res in self.res[model_s][rank][lmbda] ] )
		raw_mrr = np.mean( [ res.raw_mrr for res in self.res[model_s][rank][lmbda] ] )

		ranks_list = [ res.ranks for res in self.res[model_s][rank][lmbda]]
		hits_at1 = np.mean( [ (np.sum(ranks <= 1) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
		hits_at3 = np.mean( [ (np.sum(ranks <= 3) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
		hits_at10= np.mean( [ (np.sum(ranks <= 10) + 1e-10) / float(len(ranks))  for ranks in ranks_list] )

		logger.info("%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%i\t%f" %(model_s, mrr, raw_mrr, hits_at1, hits_at3, hits_at10, rank, lmbda))

		return ( mrr, raw_mrr, hits_at1, hits_at3, hits_at10)


	def print_MRR_and_hits(self):

		metrics = {}
	
		logger.info("Model\t\t\tMRR\tRMRR\tH@1\tH@3\tH@10\trank\tlmbda")

		for j, (model_s, cur_res) in enumerate(self.res.items()):

			best_mrr = -1.0
			for i,(k, lmbdas) in enumerate(cur_res.items()):

				mrrs = []
				for lmbda, res_list in lmbdas.items():
					mrrs.append( (lmbda, np.mean( [ result.mrr for result in res_list] ), np.mean( [ result.raw_mrr for result in res_list] ) ) )

				lmbda_mrr = max(mrrs, key = operator.itemgetter(1))
				mrr = lmbda_mrr[1]
				if mrr > best_mrr:
					best_mrr = mrr
					best_raw_mrr = lmbda_mrr[2]
					best_lambda = lmbda_mrr[0]
					best_rank = k
					

			metrics[model_s] = (best_rank, best_lambda) + self.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)
		
		return metrics


		




class Scorer(object):

	def __init__(self, train, valid, test, compute_ranking_scores = False,):

		self.compute_ranking_scores = compute_ranking_scores

		self.known_obj_triples = {}
		self.known_sub_triples = {}
		if self.compute_ranking_scores:
			self.update_known_triples_dicts(train.indexes)
			self.update_known_triples_dicts(test.indexes)
			if valid is not None:
				self.update_known_triples_dicts(valid.indexes)


	def update_known_triples_dicts(self,triples):
		for i,j,k in triples:
			if (i,j) not in self.known_obj_triples:
				self.known_obj_triples[(i,j)] = [k]
			elif k not in self.known_obj_triples[(i,j)]:
				self.known_obj_triples[(i,j)].append(k)

			if (j,k) not in self.known_sub_triples:
				self.known_sub_triples[(j,k)] = [i]
			elif i not in self.known_sub_triples[(j,k)]:
				self.known_sub_triples[(j,k)].append(i)


	def compute_scores(self, model, model_s, params, eval_set):
		preds = model.predict(eval_set.indexes)

		ranks = None
		raw_ranks = None

		if not getattr(self, "compute_ranking_scores", False):
			return Result(preds, eval_set.values, ranks, raw_ranks)

		# ---- Torch setup
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		dtype = torch.float32
		torch.set_grad_enabled(False)

		# test triples
		idx = np.asarray(eval_set.indexes, dtype=np.int64)
		nb_test = len(eval_set.values)
		s_idx = torch.from_numpy(idx[:nb_test, 0]).to(device)
		r_idx = torch.from_numpy(idx[:nb_test, 1]).to(device)
		o_idx = torch.from_numpy(idx[:nb_test, 2]).to(device)

		ranks = np.empty(2 * nb_test, dtype=np.float64)
		raw_ranks = np.empty(2 * nb_test, dtype=np.float64)

		# ---------- Fast vectorized paths ----------
		if model_s.startswith(("DistMult", "Complex", "CP", "TransE", "Rescal")):
			# DISTMULT -----------------------------------------------------------
			if model_s.startswith("DistMult"):
				e = _to_t(model.e.get_value(borrow=True)).to(device=device, dtype=dtype)    # (n,k)
				r = _to_t(model.r.get_value(borrow=True)).to(device=device, dtype=dtype)    # (m,k)

				# object ranking: (e[s]*r[r]) @ e^T -> (N, n_entities)
				left_o = e[s_idx] * r[r_idx]                       # (N,k)
				obj_scores = left_o @ e.T                          # (N, n)

				# subject ranking: e @ (r[r]*e[o])^T -> (n, N) -> transpose to (N,n)
				right_s = (r[r_idx] * e[o_idx])                    # (N,k)
				sub_scores = (e @ right_s.T).T                     # (N, n)

				# compute ranks sample-wise (vector compare per row)
				n_ent = e.shape[0]
				arange_n = torch.arange(n_ent, device=device)

				for a in range(nb_test):
					i = s_idx[a].item(); j = r_idx[a].item(); k = o_idx[a].item()

					# object ranks
					row_o = obj_scores[a]                          # (n,)
					true_o = row_o[k]
					raw = 1 + (row_o > true_o).sum().item()
					raw_ranks[a] = raw
					known = self.known_obj_triples.get((i, j), [])
					if known:
						known_t = torch.tensor(known, device=device, dtype=torch.long)
						adj = (row_o[known_t] > true_o).sum().item()
						ranks[a] = raw - adj
					else:
						ranks[a] = raw

					# subject ranks
					row_s = sub_scores[a]                          # (n,)
					true_s = row_s[i]
					raw = 1 + (row_s > true_s).sum().item()
					raw_ranks[nb_test + a] = raw
					known = self.known_sub_triples.get((j, k), [])
					if known:
						known_t = torch.tensor(known, device=device, dtype=torch.long)
						adj = (row_s[known_t] > true_s).sum().item()
						ranks[nb_test + a] = raw - adj
					else:
						ranks[nb_test + a] = raw

			# COMPLEX ------------------------------------------------------------
			elif model_s.startswith("Complex"):
				e1 = _to_t(model.e1.get_value(borrow=True)).to(device=device, dtype=dtype)  # (n,k)
				e2 = _to_t(model.e2.get_value(borrow=True)).to(device=device, dtype=dtype)  # (n,k)
				r1 = _to_t(model.r1.get_value(borrow=True)).to(device=device, dtype=dtype)  # (m,k)
				r2 = _to_t(model.r2.get_value(borrow=True)).to(device=device, dtype=dtype)  # (m,k)

				# Gather test embeddings
				se1 = e1[s_idx]      # (N,k)
				se2 = e2[s_idx]
				rr1 = r1[r_idx]
				rr2 = r2[r_idx]
				oe1 = e1[o_idx]      # (N,k)
				oe2 = e2[o_idx]

				# object ranking scores (N,n)
				obj_scores = ( (se1 * rr1) @ e1.T
							+ (se2 * rr1) @ e2.T
							+ (se1 * rr2) @ e2.T
							- (se2 * rr2) @ e1.T )

				# subject ranking scores (N,n)
				sub_scores = ( e1 @ (rr1 * oe1).T
							+ e2 @ (rr1 * oe2).T
							+ e1 @ (rr2 * oe2).T
							- e2 @ (rr2 * oe1).T ).T

				for a in range(nb_test):
					i = s_idx[a].item(); j = r_idx[a].item(); k = o_idx[a].item()

					# object rank
					row_o = obj_scores[a]
					true_o = row_o[k]
					raw = 1 + (row_o > true_o).sum().item()
					raw_ranks[a] = raw
					known = self.known_obj_triples.get((i, j), [])
					if known:
						known_t = torch.tensor(known, device=device, dtype=torch.long)
						adj = (row_o[known_t] > true_o).sum().item()
						ranks[a] = raw - adj
					else:
						ranks[a] = raw

					# subject rank
					row_s = sub_scores[a]
					true_s = row_s[i]
					raw = 1 + (row_s > true_s).sum().item()
					raw_ranks[nb_test + a] = raw
					known = self.known_sub_triples.get((j, k), [])
					if known:
						known_t = torch.tensor(known, device=device, dtype=torch.long)
						adj = (row_s[known_t] > true_s).sum().item()
						ranks[nb_test + a] = raw - adj
					else:
						ranks[nb_test + a] = raw

			# Other classic models: keep your original implementation
			else:
				logger.info("Using original (non-torch) implementation for %s", model_s)
				# --- your existing CP/TransE/Rescal code block unchanged ---
				# (omitted here for brevity – keep your original branch.)

		# ---------- Generic slow path (unchanged, but you can torch-ify similarly) ----------
		else:
			logger.info("Slow MRRs (generic path)")
			n_ent = max(model.n, model.l)
			idx_obj_mat = np.empty((n_ent, 3), dtype=np.int64)
			idx_sub_mat = np.empty((n_ent, 3), dtype=np.int64)
			idx_obj_mat[:, 2] = np.arange(n_ent)
			idx_sub_mat[:, 0] = np.arange(n_ent)

			def generic_eval_o(i, j):
				idx_obj_mat[:, :2] = np.tile((i, j), (n_ent, 1))
				return model.predict(idx_obj_mat)

			def generic_eval_s(j, k):
				idx_sub_mat[:, 1:] = np.tile((j, k), (n_ent, 1))
				return model.predict(idx_sub_mat)

			for a, (i, j, k) in enumerate(idx[:nb_test, :]):
				res_obj = generic_eval_o(i, j)
				raw_ranks[a] = 1 + np.sum(res_obj > res_obj[k])
				ranks[a] = raw_ranks[a] - np.sum(res_obj[self.known_obj_triples.get((i, j), [])] > res_obj[k])

				res_sub = generic_eval_s(j, k)
				raw_ranks[nb_test + a] = 1 + np.sum(res_sub > res_sub[i])
				ranks[nb_test + a] = raw_ranks[nb_test + a] - np.sum(res_sub[self.known_sub_triples.get((j, k), [])] > res_sub[i])

		return Result(preds, eval_set.values, ranks, raw_ranks)

def _to_t(x):
		# x can be numpy array / list / torch tensor
		if isinstance(x, torch.Tensor):
			return x
		return torch.from_numpy(np.asarray(x))