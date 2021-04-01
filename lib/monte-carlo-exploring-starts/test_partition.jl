# @Created by: OctaveOliviers
# @Created on: 2021-03-03 20:29:24
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-29 17:15:13


using Plots ; plotly()
using ProgressBars

include("mdp.jl")
include("mces.jl")
# include("utils.jl")


function plot_partition(policy_sequences, q_grid; q_star=nothing)
    """
    explain
    """
    # number of cells in partition
    cells_id = unique(policy_sequences)
    # plot partition
    p = plot()
    for id = cells_id
        q_idx = findall(x->x==id, policy_sequences)
        scatter!(hcat(q_grid(q_idx)...)[1,:], hcat(q_grid(q_idx)...)[2,:], markerstrokewidth=0)
    end
    scatter!(q_star, markershape=:xcross, color=:red)
    display(p)
end


function compute_partition(mdp, q_idx, q1_range, q2_range, q_init, num_update; seed=42)
    """
    explain
    """

    # initialise mces solver
    mces = MCES(mdp)

    # grid of q-values to evaluate
    q_grid(indices) = [[q1_range[i[1]], q1_range[i[2]]] for i=indices]

    # keep track of visited policies
    pol_vis = []
    # store sequence of policies
    pol_seq = [[] for i=1:length(q1_range), j=1:length(q2_range)]

    function idx_policy(policy)
        if policy in pol_vis
            return findall(x->x==policy, pol_vis)[1]
        else
            append!(pol_vis, [policy])
            return length(pol_vis)
        end
    end

    for i = ProgressBar(1:length(q1_range))
        for j = 1:length(q2_range)

            # initialise mces solver
            q1, q2 = q_grid([[i,j]])[1]
            q_init(mces.q, q1, q2)
            mces.q[q_idx[1]], mces.q[q_idx[2]] = q1, q2
            mces.policy = compute_policy(mdp.structure, mces.q)

            append!(pol_seq[i,j], idx_policy([i for i=1:mdp.num_sa]'*mces.policy))

            # total number of visits
            tot_vis = zeros(Float64, mdp.num_sa)

            for k = 1:num_update
                mces_exp_step!(mces, mdp, tot_vis, max_len_epi=1e2)

                append!(pol_seq[i,j], idx_policy([i for i=1:mdp.num_sa]'*mces.policy))
       
            end
        end
    end

    # plot partition
    plot_partition(pol_seq, q_grid, q_star=(mdp.q[q_idx[1]], mdp.q[q_idx[2]]))

    return 0
end


num_sa = 40
num_s = 10
mdp = generate_mdp(num_sa, num_s, seed=13)

q_idx = findall(!iszero, ([i for i=1:mdp.num_sa].*mdp.structure)[:,1])[end-1:end]
q_min = -20
q_max = 20
q_prec = 2
q1_range = q_min:q_prec:q_max
q2_range = q_min:q_prec:q_max
q_init(q, q1, q2) = q1*ones(size(q))
num_update = 5

partition = compute_partition(mdp, q_idx, q1_range, q2_range, q_init, num_update)


# @info "results" mces.q mdp.q