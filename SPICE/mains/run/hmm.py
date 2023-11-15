import bayesian_hmm
from hmmlearn import hmm
from scipy.stats import poisson
import seaborn as sns
import warnings
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                         [0.3, 0.5, 0.2, 0.0],
                         [0.0, 0.3, 0.5, 0.2],
                         [0.2, 0.0, 0.2, 0.6]])

    lambdas = np.array([0.0,10.0, 20.0, 30.0])
    lambdas = lambdas[:,np.newaxis]

    # Build an HMM instance and set parameters
    gen_model = hmm.PoissonHMM(n_components=4)
    #n_components are the number of states, which can be taken to be
    #the number of emitters, if they are indistinguishable (same lambda)
    #and we are pooling the ROI into a "single detector element"

    gen_model.startprob_ = startprob
    gen_model.transmat_ = transmat
    gen_model.lambdas_ = lambdas

    # Generate samples
    X, Z = gen_model.sample(500)
    fig,ax=plt.subplots(1,2)
    ax[0].plot(np.squeeze(X))
    ax[1].plot(np.squeeze(Z))

    scores = list()
    models = list()
    for n_components in range(1, 5):
        for idx in range(10):  # ten different random starting states
            # define our hidden Markov model
            model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                                   n_iter=10)
            model.fit(X)
            models.append(model)
            scores.append(model.score(X))
            print(f'Converged: {model.monitor_.converged}\t\t'
                  f'Score: {scores[-1]}')

    # get the best model
    model = models[np.argmax(scores)]
    print(f'The best model had a score of {max(scores)} and '
          f'{model.n_components} components')

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    states = model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(model.lambdas_[states], ".-", ms=6, mfc="orange")
    ax.plot(X)
    ax.set_title('States compared to generated')
    ax.set_xlabel('State')

    prop_per_state = model.predict_proba(X).mean(axis=0)

    bins = sorted(np.unique(X))
    fig, ax = plt.subplots()
    ax.hist(X, bins=bins, density=True)
    ax.plot(bins, poisson.pmf(bins, model.lambdas_).T @ prop_per_state)
    ax.set_title('Histogram with Fitted Poisson States')
    ax.set_xlabel('X')
    ax.set_ylabel('Proportion')
    #plt.show()

    """
    # initialise object with overestimate of true number of latent states
    hmm = bayesian_hmm.HDPHMM(X, sticky=False)
    hmm.initialise(k=10)

    results = hmm.mcmc(n=200, burn_in=10, ncores=3, save_every=10, verbose=False)
    print(results['chain_loglikelihood'])
    state_count = results['state_count']
    map_index = results['chain_loglikelihood'].index(min(results['chain_loglikelihood']))
    parameters_map = results['parameters'][map_index] #MAP parameters

    p_transition = parameters_map['p_transition']
    p_emission = parameters_map['p_emission']
    p_initial = parameters_map['p_initial']

    x = list(p_initial.keys())
    y = list(p_initial.values())
    print(state_count)
    #hmm.print_probabilities()
    """
