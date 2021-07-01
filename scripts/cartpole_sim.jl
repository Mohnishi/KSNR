"""
This file is for executing CartpoleStable task with CEM.
Spectrum radius as a cost.
"""

#------------------Packages-------------------#
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using ElasticArrays
using LyceumBase.Tools, LyceumBase, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays
using Distances


include("../utils/weightmat.jl")
include("../mujoco_models/cartpole_stable.jl")
include("../planner/PolicySelect-GT.jl")
include("../planner/PolicySelect-SP.jl")
#-----------------------Constants -------------------#
numfeatures = 50      # dimH  (for state x)
T           = 100      # task horizon (H in the note)
samplecem   = 200      # sample number of CEM every iteration
elitenum    = 20       # Elite number of CEM
cemiter     = 100        # number of iterations for CEM


@info "CartpoleStable Task"
@info "Seed number" seednum
Random.seed!(seednum)
#----------------------Environment-------------------#
mjenvs = tconstruct(CartpoleStable, Threads.nthreads());
env = CartpoleStable()
dobs, dact = length(obsspace(env)), length(actionspace(env))
dpolicyparam = 100       # dim Theta: dimension for policy

PolicyMat  = PredictionMat(dpolicyparam, dact)  # Policy is Theata pi(x) and this is pi

@info "Dimensions of observations/actions/outputs" dobs dact
@info "Simulation timestep:" LyceumBase.timestep(env)
#-----------------------Functions -------------------#
# large cost on unstable system
function lambdacost(Y::AbstractArray,X::AbstractArray,arg)
    svdx = svd(X)
    U = svdx.U
    V = svdx.V
    S = diagm(svdx.S)
    specradius = maximum(abs.(eigvals(U'*Y*V*S/(S.^2+I))))
    return 1e4 * max(1., specradius)
end
# getting action using policy feature
function getaction!(
    action::AbstractVector,
    obs,
    featurize_pi,
    PolicyMatW::AbstractArray;
) 
    mul!(action, PolicyMatW, featurize_pi(obs))
    return action
end
#------------------Define Features-------------------#
include("../utils/rff.jl")
rffbandwidth_x = 2.
rffbandwidth_pi = 2.
const rff_x = LinearRandom{Float64}(rffbandwidth_x, dobs, numfeatures)
const rff_pi = RandomFourierFunctions{Float64}(rffbandwidth_pi, dobs, dpolicyparam)
#------------------Environment-------------------#
env_tconstructor = n -> tconstruct(CartpoleStable, n)
arg = 0

#------------------Strategy Struct-------------------#
gt_polselect = PolicySelectGT(
        env_tconstructor = n -> tconstruct(CartpoleStable, n),
        phi_pol = rff_pi,      
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,   
        gamma = 1.,   #gamma  
    )

sp_polselect = PolicySelectSP(
        env_tconstructor = n -> tconstruct(CartpoleStable, n),
        phi_pol = rff_pi, 
        phi_x  =  rff_x,  
        arg = arg,     
        lambdacost = lambdacost,
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,  
        featsize_x = numfeatures, 
        gamma = 1.,   #gamma  
    )

#------------------Only Cumulative cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(gt_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi, PolicyMat.W),
            env; T = T, plotiter = 50)
for _ in opt # runs iterator
end
@info "cumulative reward" sum(opt.trajectory.rewards)

Odata = opt.trajectory.observations
@info "median value" sqrt(median(pairwise(Euclidean(), Odata, dims=2).^2)/2)
Koopman_gt = rff_x(Odata[:,2:end])*(rff_x(Odata[:,1:end-1]))'/
(rff_x(Odata[:,1:end-1])*(rff_x(Odata[:,1:end-1]))'+I)
Lcost_gt = lambdacost(rff_x(Odata[:,2:end]),rff_x(Odata[:,1:end-1]),arg)
@info "spectrum cost" Lcost_gt


#------------------With Koopman cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(sp_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt_sp = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi, PolicyMat.W),
                                 env; T = T, plotiter = 50)
for _ in opt_sp # runs iterator
end
@info "cumulative cost" sum(opt_sp.trajectory.rewards)

Odata_sp = opt_sp.trajectory.observations
Koopman_sp = rff_x(Odata_sp[:,2:end])*(rff_x(Odata_sp[:,1:end-1]))'/
(rff_x(Odata_sp[:,1:end-1])*(rff_x(Odata_sp[:,1:end-1]))'+I)
Lcost_sp = lambdacost(rff_x(Odata_sp[:,2:end]),rff_x(Odata_sp[:,1:end-1]),arg)
@info "spectrum cost" Lcost_sp

#------------------Savind data-------------------#
exper = Experiment("log/data2.jlso", overwrite = true)
exper[:rff_x] = rff_x
exper[:rff_pi] = rff_pi

exper[:Koopman_gt] = Koopman_gt; exper[:Lcost_sp] = Lcost_sp; 
exper[:Lcost_gt] = Lcost_gt;
exper[:opt_sp] = opt_sp; exper[:opt] = opt
exper[:Koopman_sp] = Koopman_sp;
exper[:cumulative_gt] = sum(opt.trajectory.rewards); 
exper[:cumulative_sp] = sum(opt_sp.trajectory.rewards)

finish!(exper);
