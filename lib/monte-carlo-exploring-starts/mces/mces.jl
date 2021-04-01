# @Created by: OctaveOliviers
# @Created on: 2021-01-15 17:36:10
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 14:53:56


mutable struct MCES
    """
    struct for Monte Carlo Exploring Starts solver
    """

    q::Vector
    policy::Matrix
    prior::Vector

    function MCES(
        mdp::MDP;
        seed::Integer=NO_SEED
        )
        """
        constructor
        """
        # set random number generator
        if seed != NO_SEED; Random.seed!(seed); end

        # initialise q-values
        q = rand(mdp.num_sa)
        # q = [5, 5.1, 0, 0]
        # initialise policy
        policy = compute_policy(mdp.structure, q)
        # initialise prior
        prior = normalize(rand(mdp.num_sa), 1)

        new(q, policy, prior)
    end
end # struct MCES
