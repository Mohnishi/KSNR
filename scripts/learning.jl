"""
This file is for executing Gym cartpole task with learning.
Note this is for decomposable kernel only.
"""

#------------------Packages-------------------#
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using UnicodePlots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays, ElasticArrays
using Distances


include("../utils/weightmat.jl")
include("../models/cartpole.jl")
include("../planner/MPPIClamp.jl")

#-----------------------Constants -------------------#
numfeatures = 60      # dimH  (for state x)
trainT      = 100       # training horizon for getting target koopman
trainiter   = 20

@info "CartpoleGym Task"
seednum = 100
@info "Seed number" seednum
Random.seed!(seednum)
#----------------------Environment-------------------#
mjenvs = tconstruct(CartpoleGym, Threads.nthreads());
env = CartpoleGym()
dobs, dact = length(obsspace(env)), length(actionspace(env))
dpolicyparam = 3      # dim Theta: dimension for policy
dpolibase  = 2000
ctrlrange = env.ctrlrange
#------------------Define Functions-------------------#
# getting ridge regressor
function getpred(Y::AbstractArray,X::AbstractArray,reg::AbstractFloat)
        return Y * X' / (X * X' + reg*I)
end
# update target koopman data
function updatedata!(OX::AbstractArray, OY::AbstractArray,
    A::AbstractArray, env::AbstractEnvironment,
    planner, tT::Integer)
    train = ControllerIterator((action, state, obs) -> LyceumBase.getaction!(action, state, planner),
                env; T = tT, plotiter = tT)
    for _ in train # runs iterator
    end
    Odata_train = train.trajectory.observations
    Adata_train = train.trajectory.actions
    append!(OX, Odata_train[:,1:end-1])
    append!(OY, Odata_train[:,2:end])
    append!(A, Adata_train[:,1:end-1])
end
# train target kooman and policies
function trainKoop(iter::Integer, trainT::Integer, planner)
    Odata_trainX1 = ElasticArray{Float64}(undef,dobs,0)
    Odata_trainY1 = ElasticArray{Float64}(undef,dobs,0)
    Adata_train1 = ElasticArray{Float64}(undef,dact,0)
    Odata_trainX2 = ElasticArray{Float64}(undef,dobs,0)
    Odata_trainY2 = ElasticArray{Float64}(undef,dobs,0)
    Adata_train2 = ElasticArray{Float64}(undef,dact,0)
    Odata_trainX = ElasticArray{Float64}(undef,dobs,0)
    Odata_trainY = ElasticArray{Float64}(undef,dobs,0)
    Adata_train0 = ElasticArray{Float64}(undef,dact,0)
    for i=1:trainiter
        @info "train iteration" i
            # going to left
            randreset!(env); env.state[5] = 1.; reset!(planner)
            updatedata!(Odata_trainX, Odata_trainY, Adata_train0,
                env, planner, trainT)
            # going to right
            env.state[5] = 2.; reset!(planner) 
            updatedata!(Odata_trainX, Odata_trainY, Adata_train0,
                env, planner, trainT+20)
            # velocity left
            randreset!(env); env.state[5] = 3.;reset!(planner)
            updatedata!(Odata_trainX1, Odata_trainY1, Adata_train1,
                env, planner, trainT)
            # velocity right
            randreset!(env); env.state[5] = 4.;reset!(planner)
            updatedata!(Odata_trainX2, Odata_trainY2, Adata_train2,
                env, planner, trainT)
    end
    Odata_trainX1 = Matrix(Odata_trainX1);Odata_trainY1 = Matrix(Odata_trainY1);
    Odata_trainX2 = Matrix(Odata_trainX2);Odata_trainY2 = Matrix(Odata_trainY2);
    Odata_trainX = Matrix(Odata_trainX); Odata_trainY = Matrix(Odata_trainY)
    errornum_train = cos.(Odata_trainX[3,:])
    @info "error number" length(errornum_train[errornum_train.<0])
    Adata_train1 = Matrix(Adata_train1);Adata_train2 = Matrix(Adata_train2)
    Adata_train0 = Matrix(Adata_train0)

    return [Adata_train0, Adata_train1, Adata_train2,
    Odata_trainX, Odata_trainX1, Odata_trainX2,
    Odata_trainY, Odata_trainY1, Odata_trainY2]
end

@info "Dimensions of observations/actions/outputs" dobs dact
@info "Simulation timestep:" LyceumBase.timestep(env)

#------------------Define Features-------------------#
include("../utils/rff.jl")
rffbandwidth_x = 1.5
rffbandwidth_pi = 1.5
const rff_x = LinearRandom{Float64}(rffbandwidth_x, dobs, numfeatures)
const rff_pi_base = RandomFourierFunctions{Float64}(rffbandwidth_pi, dobs, dpolibase)
#------------------Main setting-------------------#

env_tconstructor = n -> tconstruct(CartpoleGym, n)

# MPPI struct for training of target koopman
gt_mppi = MPPIClamp(
            env_tconstructor = n -> tconstruct(CartpoleGym, n),   #This is for GT-MPPI
            covar = Diagonal(0.4 ^2 * I, size(actionspace(env), 1)),
            lambda = 0.1,
            H = trainT,
            K = 524,
            gamma = 1.,
            clamps = ctrlrange
           )

@info "Getting pretrained policies/Koopman"
trainData =  trainKoop(trainiter, trainT, gt_mppi)
W0 = getpred(trainData[1], rff_pi_base(trainData[4]), 0.001)
W1 = getpred(trainData[2], rff_pi_base(trainData[5]), 0.001)
W2 = getpred(trainData[3], rff_pi_base(trainData[6]), 0.001)
Koop_targ = getpred(rff_x(trainData[7]), rff_x(trainData[4]), 1.)
arg = Matrix(Koop_targ)

# creating custom policy function
policyfunc = (x)->(phi = zeros(3);phi_base = rff_pi_base(x);
phi[1] = (W0 * phi_base)[1];phi[2] = (W1 * phi_base)[1];
phi[3] = (W2 * phi_base)[1]; phi) 



