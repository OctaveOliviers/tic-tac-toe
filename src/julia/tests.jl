# @Created by: OctaveOliviers
# @Created on: 2021-01-15 18:50:08
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-02 14:14:57


using ProgressBars

include("mces.jl")
include("mdp.jl")
include("utils.jl")

@debug "Set seed" Random.seed!(Dates.day(today()))

# # structure of markov tree
# S = [[1 0 0 0 0]; 
#      [1 0 0 0 0]; 
#      [0 1 0 0 0]; 
#      [0 1 0 0 0]; 
#      [0 0 1 0 0]; 
#      [0 0 1 0 0]; 
#      [0 0 0 1 0]; 
#      [0 0 0 0 1]]
# # transition probabilities
# T = [[0 0 0 0 1 0 0 0]; 
#      [1 0 0 0 0 0 0 0]; 
#      [0 0 1 0 0 0 0 0]; 
#      [0 1 0 1 0 0 1 0]; 
#      [0 0 0 0 0 1 0 1]]
# # rewards of each state-action
# r = [0, -10, 0, -10, 0, 10, 0, 0]
# # optimal q values and policy
# q_opt = [8.1, -10, 9, -10, 7.29, 10, 0, 0]
# P_opt = compute_policy(S, q_opt)

# test size of weights
if false

    # # create MDP
    # num_s = 3+convert(Int64, ceil(5*rand()))
 #    num_sa = convert(Int64, 2*num_s + ceil(20*rand()))

 #    # discount factor
 #    gam = 0.9*rand()
 #    # prior probability of starting episode in each state-action
 #    p = rand(Float64, num_sa)
 #    p = p/sum(p)

 #    S, T, P_opt, r, q_opt, gam = gen_mdp(num_sa, num_s, gam=gam)
    # @info "Parameters are" num_s num_sa P_opt'*q_opt gam

    gam = 0.9 ;
    p = [0.1, 0.2, 0.1, 0.4, 0.2, 0, 0, 0]
    S = [[1 0 0 0 0 0];
         [0 1 0 0 0 0];
         [0 1 0 0 0 0];
         [0 0 1 0 0 0];
         [0 0 1 0 0 0];
         [0 0 0 1 0 0];
         [0 0 0 0 1 0];
         [0 0 0 0 0 1]]
    T = [[0 0 0 0 0 0 0 0];
         [1 0 0 .8 0 0 0 0];
         [0 0 1 0 0 0 0 0];
         [0 1 0 0 0 1 0 0];
         [0 0 0 0 1 0 1 0];
         [0 0 0 .2 0 0 0 1]]
    r = [0., 10., 0., -10., -20., 0., 0., 0.] ;
    P = [[1 0 0 0 0 0];
         [0 1 0 0 0 0];
         [0 0 0 0 0 0];
         [0 0 1 0 0 0];
         [0 0 0 0 0 0];
         [0 0 0 1 0 0];
         [0 0 0 0 1 0];
         [0 0 0 0 0 1]]

    q = monte_carlo_exploring_start(S, T, r, p, gam, num_epi = 1e6, max_len_epi = 2*1e1)
    # q = expected_mces(S, T, r, p, gam, num_epi = 1e3, max_len_epi = 4*1e2)

    @info q
end

# test mote carlo algorithm
if true

    num_test = 10

    for n = 1:num_test
        println("\nTest ", n, " of ", num_test)

        num_s = 3+convert(Int64, ceil(5*rand()))
        num_sa = convert(Int64, 2*num_s + ceil(20*rand()))

        # discount factor
        gam = 0.9*rand()
        # prior probability of starting episode in each state-action
        p = rand(Float64, num_sa)
        p = p/sum(p)

        S, T, P_opt, r, q_opt, gam = gen_mdp(num_sa, num_s, gam=gam)

        @info "Parameters are" num_s num_sa P_opt'*q_opt gam

        # q = monte_carlo_exploring_start(S, T, r, p, gam)
        q = expected_mces(S, T, r, p, gam, num_epi = 5*1e2, max_len_epi = 4*1e2)
        P = compute_policy(S,q)

        println("Found correct policy: ", P == P_opt)
    end

    # println("\nvalues are       $(round.(P'*q, digits=3))")
    # println("\nvalues should be $(round.(P_opt'*q_opt, digits=3))")
end

# test inequalities on weights
if false

    num_sa = 30
    num_s = 10

    # discount factor
    gam = 0.9
    # prior probability of starting episode in each state-action
    p = rand(Float64, num_sa)
    p = p/sum(p)

    S, T, P_opt, r, q_opt, gam = gen_mdp(num_sa, num_s)

    q0 = 100*rand(num_sa)
    P1 = compute_policy(S, q0)
    pot, val, q1 = compute_potential(P1, T, r, p, q0, gam, 400, return_q_opt=true) ;

    P2 = compute_policy(S, q1)
    pot, val, q2 = compute_potential(P2, T, r, p, q1, gam, 400, return_q_opt=true) ; 

    W = Diagonal(rand(num_sa))
    q3 = W*q1 + (I-W)*q2

    P3 = compute_policy(S, q3)
    # pot, val, q2 = compute_potential(P3, T, r, p, q1, gam, 400, return_q_opt=true) ;

    if P3 != P2
        @info "new policy"
    end

    println( W*T'*P2'*q1 + (I-W)*T'*P2'*q2 .<= T'*P3'*(W*q1 + (I-W)*q2) )
end

# println(q)
# println("Computed q-values are $q")
# println("Should be close to $q_opt")
# println("computed vs actual q-values")
# println("q values are $(round.([q,q_opt], digits=3))")

