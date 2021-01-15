# @Author: OctaveOliviers
# @Date:   2021-01-15 18:50:08
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-15 19:25:13

include("monte_carlo_early_start.jl")

# structure of markov tree
S = [[1 0 0 0 0]; 
     [1 0 0 0 0]; 
     [0 1 0 0 0]; 
     [0 1 0 0 0]; 
     [0 0 1 0 0]; 
     [0 0 1 0 0]; 
     [0 0 0 1 0]; 
     [0 0 0 0 1]]
# transition probabilities
T = [[0 0 0 0 1 0 0 0]; 
     [1 0 0 0 0 0 0 0]; 
     [0 0 1 0 0 0 0 0]; 
     [0 1 0 1 0 0 1 0]; 
     [0 0 0 0 0 1 0 1]]
# rewards of each state-action
R = [0, -10, 0, -10, 0, 10, 0, 0]
# prior probability of starting episode in each state-action
p = rand(Float16, 8)
p = p/sum(p)
# discount factor
gam = 0.9

q = monte_carlo_early_start(S, T, R, p, gam)

# println(q)
println("Computed q-values are $q")
println("Should be close to [ 8.1 -10 9 -10 7.29 10 0 0 ]")