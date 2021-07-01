#------------------Policy Select-------------------#
@info "Running learning algorithm"
include("../utils/algorithm.jl")
include("../utils/learned_env.jl")
include("../planner/PolicySelect.jl")
#--------------#
T           = 500      # task horizon (H in the note)
numopkernel = 50      # D: #features for operator valued [decomposable kernel: psi(Theta)]
totalrank = numfeatures * numopkernel #numfeatures * Brank * numopkernel -> can be reduced to numfeatures * numopkernel
lambda_reg  = 1.   # prior covariance is I/lambda_reg
TS_scale    = 0.0001   # posterior reshaping constant for Thompson sampling
samplecem   = 200      # sample number of CEM every iteration
elitenum    = 20       # Elite number of CEM
cemiter     = 50        # number of iterations for CEM
rffbandwidth_op = 5.
#--------------#
@info "Seed number" seednum
Random.seed!(seednum)
#--------------#
mjenvs = tconstruct(CartpoleGym, Threads.nthreads());
const rff_op = RandomFourierFunctions{Float64}(rffbandwidth_op, dpolicyparam*dact, numopkernel)
PredictMat  = PredictionMat(totalrank, numfeatures) # for Bmat=I case
PolicyMat  = PredictionMat(dpolicyparam, dact)  # Policy is Theata pi(x) and this is pi
KoopmanMat = PredictionMat(numfeatures, numfeatures)  # Psi()*M

#----define functions-----#
isdonefunc(x) = (cos(x[3])<0)  
    # eliminate data that are bad
function lambdacost(K::AbstractArray,arg)
    sum((arg[1:4,:]-K[1:4,:]).^2) + 
    0.01 * sum(abs.(eigvals(K)))
end
# Reward function: feature includes obs in our case
function rewfunc(fo)
        theta = fo[3]
        v = fo[2]
        reward = 0.
        if cos(theta) < 0
                reward = -100.
        end
        reward - 0.001*(abs(v)-1.5)^2
        #reward
end
# get action from policy feature
function getaction!(
        action::AbstractVector,
        obs,
        featurize_pi,
        PolicyMatW::AbstractArray;
    ) 
        mul!(action, PolicyMatW, featurize_pi(obs))
        return action
end
#------------------Strategy Struct-------------------#
polselect = PolicySelect(
            env_tconstructor = n -> tconstruct(LearnedEnv, KoopmanMat.W, rff_x, rewfunc, mjenvs, n),
            Mmat = copy(reshape(PredictMat.W, numfeatures, numopkernel, numfeatures)),
            phi_op = rff_op, 
            lambdacost = lambdacost,    
            arg = arg,   
            samplecem = samplecem,   
            elitenum = elitenum,     
            cemiter = cemiter,       
            H = T,   #H
            Policyparamsize = dpolicyparam*dact,   
            gamma = 1.   #gamma  
           )

randreset!(env)
alg = Alg(
        env,
        rff_x,
        rff_op,
        policyfunc,
        PredictMat,
        PolicyMat,
        numfeatures,
        numopkernel,
        dpolicyparam,
        totalrank,
        dobs,
        dobs,
        dact,
        ctrlrange,
        isdonefunc,
        lambdacost,
        arg,
        lambda_reg,
        TS_scale,
        polselect;
        Hmax = T,
        N = T,
        );

#------------------Running the algo-------------------#
function cpg_alg(alg::Alg, plot::Bool; NITER=1000)
    # save data to the following file
    exper = Experiment("log/data4_"*dfile_num*".jlso", overwrite = true)

    lg = ULogger()
    for (i, state) in enumerate(alg)
        if i >= NITER
            # save some constants here (Weight matrix is at the terminal episode)
            exper[:feat_x] = alg.featurize_x
            exper[:feat_op] = alg.featurize_op
            exper[:feat_pi] = alg.featurize_pi
            exper[:timestep] = LyceumBase.timestep(env)
            exper[:PolicyMat] = alg.PolicyMat
            exper[:PredictMat] = alg.PolicyMat
            exper[:traj] = state.traj
            break
        end
        # save all the log data
        push!(lg, :algstate, filter_nt(state, exclude = (:elapsed_sampled)))
        if plot && mod(i, 1) == 0
            x = lg[:algstate]
            # show reward curve
            display(expplot(
                            Line(x[:traj_reward], "reward evaluations"),
                            title = "reward evaluations, Iter=$i",
                            width = 30, height = 7,
                           ))
            # show spectrum curve
            display(expplot(
                            Line(x[:traj_spectrum], "spectrum evaluations"),
                            title = "spectrum evaluations, Iter=$i",
                            width = 30, height = 7,
                           ))
            # show spectrum_est curve
            display(expplot(
                            Line(x[:traj_spectrum_est], "spectrum_est evaluations"),
                            title = "spectrum_est evaluations, Iter=$i",
                            width = 30, height = 7,
                           ))
            # show prediction error plots
            display(expplot(
                            Line(x[:prederr], "prediction error"),
                            title = "model prediction error, Iter=$i",
                            width = 30, height = 7,
                           ))

        end
    end
    exper, lg
end

exper, lg = cpg_alg(alg, true; NITER=20);

exper[:logs] = get(lg)
finish!(exper); # flushes everything to disk