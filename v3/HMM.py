import logging
from _thread import allocate_lock, start_new_thread
from time import sleep

import tensorboardX
from hmmlearn import _hmmc as hmmlearn_hmmc
from hmmlearn import _utils as hmmlearn_utils
from hmmlearn import hmm, base
from hmmlearn.utils import *
from sklearn.utils import check_array
from tqdm import tqdm

_log = logging.getLogger(__name__)


class TbXMonitor(base.ConvergenceMonitor):
    """HMM Monitor writing logs to tensorboard """

    def __init__(self, tol, n_iter, name, model: base._BaseHMM):
        """ Setup moitor
        :param tol: Convergence threshold. Converged if the log probability improvement
                    between two consecutive iterations is less than threshold.
        :param n_iter: maximum number of iterations to be performed
        :param name: trial name to be used in tensorboard
        :param model: model to be monitored
        """
        super(TbXMonitor, self).__init__(tol, n_iter, False)
        self.model = model
        self.log = tensorboardX.SummaryWriter("runs/" + name)
        self.iter = 0

    def report(self, logprob):
        """Reports to Tesorboard.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        # add logprob delta to log
        if self.history:
            delta = logprob - self.history[-1]
            self.log.add_scalar("delta", delta, global_step=self.iter)
        self.log.add_scalar("logprob", logprob, global_step=self.iter)
        self.history.append(logprob)  # keep track of the last two iterations

        # add matrix images to log
        self.log.add_image("transmat_", (self.model.transmat_ / self.model.transmat_.max())[None, :, :],
                           global_step=self.iter)
        self.log.add_image("startprob_", (self.model.startprob_ / self.model.startprob_.max())[None, None, :],
                           global_step=self.iter)
        self.log.add_image("emissionprob_",
                           (self.model.emissionprob_ / self.model.emissionprob_.max())[None, :, :],
                           global_step=self.iter)

        # increase iteration count
        self.iter += 1


def iter_from_X_lengths(X, lengths, desc=None):
    """ generator returning the sentences of the dataset
    :param X: Dataset containing concatenated sentences
    :param lengths: List of sentence lengths
    :param desc: Description shown in progressbar
    :return: generator
    """
    if lengths is None:  # if the length is not given only one sentence is given
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in tqdm(range(len(lengths)), desc=desc):
            yield start[i], end[i]  # return start and end position of sentence


class MultiThreadFit(hmm.MultinomialHMM):
    """ Multi threaded version of the MultinomialHMM"""

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste", num_workers=8):
        super(MultiThreadFit, self).__init__(n_components=n_components,
                                             startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                                             algorithm=algorithm, random_state=random_state,
                                             n_iter=n_iter, tol=tol, verbose=verbose,
                                             params=params, init_params=init_params)
        self.num_workers = num_workers

    class ThreadsafeIter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """

        def __init__(self, it):
            self.it = iter(it)
            self.lock = allocate_lock()

        def __iter__(self):
            return self

        def next(self):
            with self.lock:
                return next(self.it)

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        # Code from hmmlearn modified to be thread save
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """

        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            hmmlearn_hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                              log_mask_zero(self.transmat_),
                                              bwdlattice, framelogprob,
                                              log_xi_sum)

        with self.stats_lock:  # ensure only one thread at a time changes the stats -> prevent race conditions
            stats['nobs'] += 1
            if 's' in self.params:
                stats['start'] += posteriors[0]
            if 't' in self.params:
                with np.errstate(under="ignore"):
                    stats['trans'] += np.exp(log_xi_sum)
            if 'e' in self.params:
                for t, symbol in enumerate(np.concatenate(X)):
                    stats['obs'][:, symbol] += posteriors[t]

    def fit(self, X, lengths=None):
        # Code from hmmlearn modified to be multi threaded
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)

        if not self.monitor_.iter:  # initial checks and setup if it is the first iteration
            self._init(X, lengths=lengths)
            self._check()

        for iter in range(self.n_iter):  # fit model for given number of iterations
            stats = self._initialize_sufficient_statistics()
            self.curr_logprob = 0

            iterator = self.ThreadsafeIter(iter_from_X_lengths(X, lengths, desc=f"Fitting {self.monitor_.iter}"))
            thread_exit_locks = []
            self.stats_lock = allocate_lock()
            logprob_lock = allocate_lock()

            for i in range(self.num_workers):  # generate workers to process the sentences
                thread_exit_locks.append(allocate_lock())
                start_new_thread(self.fit_thread, (), {"model": self,
                                                       "iterator": iterator,
                                                       "stats": stats,
                                                       'logprob_lock': logprob_lock,
                                                       "X": X,
                                                       "exit_lock": thread_exit_locks[-1]})

            sleep(1)  # wait for all threads to start
            for i, t in enumerate(thread_exit_locks):  # wait for the threads to finish
                with t:  # try to acquire lock held by thread -> if acquired, thread is signaling end of work
                    pass

            self.stats_lock = None  # keep model dumpable with pickle

            self._do_mstep(stats)
            self.monitor_.report(self.curr_logprob)
            del self.curr_logprob
            if self.monitor_.converged:
                break

        if (self.transmat_.sum(axis=1) == 0).any():
            _log.warning("Some rows of transmat_ have zero sum because no "
                         "transition from the state was ever observed.")

        return self

    def fit_thread(self, model, stats, iterator, logprob_lock, X, exit_lock):
        # Code from hmmlearn modified to be multi threaded
        """ Code executed by a thread fitting the model to the data
        :param model: model to be fitted
        :param stats: stats to be accumulated
        :param iterator: thread save generator providing the sentences to be fitted to
        :param logprob_lock: lock preventing race conditions on accumulated logprob
        :param X: train data
        :param exit_lock: lock used to signal finished working
        """
        with exit_lock:  # hold lock acquired to signal work in progress
            while True:  # run until all sentences are processed
                try:
                    i, j = iterator.next()  # get next sentence
                except StopIteration:  # generator signals end of sentences
                    break  # finish

                # fix model to sentence
                framelogprob = model._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = model._do_forward_pass(framelogprob)
                with logprob_lock:  # accumulate with one threat at a time
                    model.curr_logprob += logprob
                bwdlattice = model._do_backward_pass(framelogprob)
                posteriors = model._compute_posteriors(fwdlattice, bwdlattice)
                model._accumulate_sufficient_statistics(stats, X[i:j], framelogprob, posteriors, fwdlattice, bwdlattice)

    def score(self, X, lengths=None):
        # Code from hmmlearn modified to be multi threaded
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        hmmlearn_utils.check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        self.logprob = 0

        iterator = self.ThreadsafeIter(iter_from_X_lengths(X, lengths, desc="Score"))
        thread_exit_locks = []
        logprob_lock = allocate_lock()

        for i in range(self.num_workers):  # generate workers to process the sentences
            thread_exit_locks.append(allocate_lock())
            start_new_thread(self.score_thread, (), {"model": self,
                                                     "iterator": iterator,
                                                     'logprob_lock': logprob_lock,
                                                     "X": X,
                                                     "exit_lock": thread_exit_locks[-1]})

        sleep(1)  # wait for all threads to start
        for i, t in enumerate(thread_exit_locks):  # wait for the threads to finish
            with t:  # try to acquire lock held by thread -> if acquired, thread is signaling end of work
                pass

        return self.logprob

    def score_thread(self, model, iterator, X, logprob_lock, exit_lock):
        # Code from hmmlearn modified to be multi threaded
        """ Code executed by a thread scoring the model
        :param model: model to be scored
        :param iterator: thread save generator providing the sentences to be fitted to
        :param logprob_lock: lock preventing race conditions on accumulated logprob
        :param X: train data
        :param exit_lock: lock used to signal finished working
        """
        with exit_lock:  # hold lock acquired to signal work in progress
            while True:  # run until all sentences are processed
                try:
                    i, j = iterator.next()  # get next sentence
                except StopIteration:  # generator signals end of sentences
                    break  # finish

                # score model on sentence
                framelogprob = model._compute_log_likelihood(X[i:j])
                logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
                with logprob_lock:  # accumulate logprob with one thread at a time
                    self.logprob += logprobij
