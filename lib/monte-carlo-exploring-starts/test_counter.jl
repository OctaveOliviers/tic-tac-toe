# @Created by: OctaveOliviers
# @Created on: 2021-03-03 20:29:24
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 14:53:57

using Plots ; plotly()
using ProgressBars

include("mdp.jl")
include("mces.jl")


# define MDP
begin
    gam = 0.9 ;
    S = [[1 0 0];
         [1 0 0];
         [0 1 0];
         [0 0 1]]
    T = [[0.9 0.8 0 0];
         [0.1 0 1 0];
         [0 0.2 0 1]]
    r = [1.1, 1.2, 0, 0] ;
    P = [[1 0 0];
         [0 0 0];
         [0 1 0];
         [0 0 1]]
    q = compute_q_policy(P, T, r, gam)

    mdp = MDP(size(S,2), size(S,1), gam, S, P, T, r, q, [3,4])
end

# initialise mces solver
mces = MCES(mdp, seed=13)


### General episode-wise MCES
# run_mces!(mces, mdp, num_epi=1e6)


### Expected MCES
# run_mces_exp!(mces, mdp, num_epi=1e5)


num_epi = 1e5
p=plot()

for q0 = []
    # total number of visits
    tot_vis = zeros(Float64, mdp.num_sa)

    # initialise MCES solver
    mces.q[1:2] = 
    mces.policy = compute_policy(mdp.structure, mces.q)

    path = []
    append!(path, [mces.q[1:2]])

    for k = ProgressBar(1:num_epi)
        # change prior of mces solver
        p_max = 0.99
        if mces.q[1,1] == 1
            mces.prior = [ p_max, (1-p_max)/3, (1-p_max)/3, (1-p_max)/3]
        else
            mces.prior = [ (1-p_max)/3, p_max, (1-p_max)/3, (1-p_max)/3]
        end

        # perform one mces expected step
        # mces_exp_step!(mces, mdp, tot_vis, max_len_epi=1e2)
        epi_stp = compute_q_policy(mces.policy, mdp.transitions, mdp.rewards, mdp.discount) - mces.q

        # update q-estimates
        mces.q += epi_stp.*mces.prior/k
        # compute policy matrix from q-values
        mces.policy = compute_policy(mdp.structure, mces.q)

        append!(path, [mces.q[1:2]])

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< TOLERANCE)
            break
        end
    end

    stp = 1e2
    plot!(hcat(path...)[1,1:stp:end], hcat(path...)[2,1:stp:end])
end
scatter!((mdp.q[1], mdp.q[2]), markershape=:xcross, color=:red)
plot!(4:6, 4:6)
display(p)

@info "results" mces.q mdp.q