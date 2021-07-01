using Base.Iterators: partition
using ElasticArrays
using LinearAlgebra
using LyceumAI
using Kronecker
BLAS.set_num_threads(32)
# data buffer
struct DataBuffers{T<:AbstractFloat}
    phixdatabuf::ElasticArray{T}      # phi(x_t)
    phiydatabuf::ElasticArray{T}      # phi(x_y)
    opphidatabuf::ElasticArray{T}      # Phi(policyparam) ...
    ydatabuf::ElasticArray{T}          # x_t+1
    xdatabuf::ElasticArray{T}          # x_t
    udatabuf::ElasticArray{T}          # u_t
    rewarddatabuf::ElasticArray{T}     # reward_t
    kspectrumdatabuf::ElasticArray{T}  # koopman mat spectrum
    sizeofbuf::ElasticArray{Int}       # size of the buffer
    function DataBuffers{T}(no::Int, nu::Int, nfx::Int, nfop::Int) where T<:AbstractFloat
        new(ElasticArray{T}(undef,nfx,0),
            ElasticArray{T}(undef,nfx,0),
            ElasticArray{T}(undef,nfx*nfop,0),
            ElasticArray{T}(undef,no,0),
            ElasticArray{T}(undef,no,0),
            ElasticArray{T}(undef,nu,0),
            ElasticArray{T}(undef,1,0),
            ElasticArray{T}(undef,1,0),
            ElasticArray{Int}(undef,1,0)
           )
    end
end

# Alg structure
struct Alg{E, FA, FB, FC, K, S, D, L, A}
    envsampler::E
    polselect::S
    featurize_x::FA
    featurize_op::FB
    featurize_pi::FC
    PredictMat::K
    PolicyMat::K
    Hmax::Int      # maximum horizon
    N::Int         # number of trajectories
    ctrlrange::AbstractArray
    isdonefunc::D
    lambdacost::L
    arg::A
    covW::AbstractArray
    Amat::AbstractArray
    covscale::AbstractFloat
    numfeatures::Int
    numopkernel::Int
    buffers::DataBuffers    # replay buffer sort of..
    function Alg(
                 env_tconstructor,
                 featurize_x,
                 featurize_op,
                 featurize_pi,
                 PredictMat,
                 PolicyMat,
                 numfeatures,
                 numopkernel,
                 dpolicyparam,
                 totalrank,
                 ns,
                 no,
                 nu,
                 ctrlrange,
                 isdonefunc,
                 lambdacost,
                 arg,
                 ridgereg,
                 covscale,
                 polselect;

                 Hmax = 400, 
                 N = 400,
                 prebuffer = nothing,
                )

        # check errors
        0 < Hmax <= N || throw(ArgumentError("Hmax must be in interval (0, N]"))
        0 < N || throw(ArgumentError("N must be > 0"))
        # find common type of PredictionMat
        envsampler = env_tconstructor 
        new{
            typeof(envsampler),
            typeof(featurize_x),
            typeof(featurize_op),
            typeof(featurize_pi),
            typeof(PredictMat),
            typeof(polselect),
            typeof(isdonefunc),
            typeof(lambdacost),
            typeof(arg),
           }(
             envsampler,
             polselect,
             featurize_x,
             featurize_op,
             featurize_pi,
             PredictMat,
             PolicyMat,
             Hmax,
             N,
             ctrlrange,
             isdonefunc,
             lambdacost,
             arg,
             Matrix(I(totalrank)/ridgereg),
             zeros(totalrank, numfeatures),
             covscale,
             numfeatures,
             numopkernel,
             prebuffer == nothing ? DataBuffers{Float32}(no,nu,numfeatures,numopkernel) : prebuffer,
            )
    end
end

# getaction with policyvec
function getaction!(
    action::AbstractVector,
    state,
    featurize_pi,
    PolicyMatW::AbstractArray;
) 
    mul!(action, PolicyMatW, featurize_pi(state))
    return action
end

#Get coherent data from multiple trajectory data
function getshifteddata(trajectory::AbstractArray)
    data1 = trajectory[:,1:end-1]
    data2 = trajectory[:,2:end]
    return (data1, data2)
end

#Clamp control inputs to the given range
function clampctrl!(actions::AbstractArray, ctrlrange::AbstractArray)
    nu = size(actions, 1)
    @inbounds for u=1:size(ctrlrange, 2)
        @simd for a=1:nu
            actions[u,a] = clamp(actions[u,a], ctrlrange[1,u], ctrlrange[2,u])
        end
    end
end

#Storing data and plot
function storedata!(batch::NamedTuple, buffers::DataBuffers, ctrlrange::AbstractArray,
    featurize_x, featurize_op, PolicyMat, Mmat::AbstractArray, isdonefunc, lambdacost,
    arg)
    # Storing data into arrays : getshifteddata deals with multiple trajecotries
    isdonei = size(batch.observations, 2)
    for i in 1:size(batch.observations, 2)
        if isdonefunc(batch.observations[:,i]) 
            isdonei = i-1
            break
        end
    end
    @info "isdone" isdonei
    xydata = getshifteddata(batch.observations[:,1:isdonei])
    uvdata = getshifteddata(batch.actions[:,1:isdonei])
    xdata  = first(xydata) 
     
    medx  = sqrt(median(pairwise(Euclidean(), xdata, dims=2).^2)/2)
    println("pairwise dist median: ", medx)

    ydata  = last(xydata)
    udata  = first(uvdata) 
    rewarddata = batch.rewards[1:isdonei-1]
    clampctrl!(udata, ctrlrange)
    PhiXdata = featurize_x(xdata)
    PhiYdata  = featurize_x(ydata) #- PhiXdata
    PhiforK  = featurize_op(reshape(PolicyMat.W, length(PolicyMat.W), 1))
    KoopMat = reduce(hcat, [Mmat[:,:,i]*PhiforK for i in 1:size(Mmat,3)])
    #---------#
    Koop_inst = PhiYdata * PhiXdata' / (PhiXdata*PhiXdata'+I)
    Kspectrum  = lambdacost(Koop_inst, arg)
    Kspectrum_est  = lambdacost(KoopMat, arg)
    OpPhidata = Matrix(PhiXdata[:,:] ⊗ PhiforK)
    # save data into data buffers
    append!(buffers.phixdatabuf, PhiXdata)
    append!(buffers.phiydatabuf, PhiYdata)
    append!(buffers.opphidatabuf, OpPhidata)
    append!(buffers.ydatabuf, ydata) 
    append!(buffers.xdatabuf, xdata) 
    append!(buffers.udatabuf, udata)
    append!(buffers.rewarddatabuf, rewarddata) 
    append!(buffers.kspectrumdatabuf, Kspectrum) 
    append!(buffers.sizeofbuf, [size(xdata,2)])  # data size 
    return PhiXdata, PhiYdata, OpPhidata, rewarddata, Kspectrum, Kspectrum_est
end


#Iterator for PredictionMat Learning
function Base.iterate(featlearn::Alg{DT}, i = 1) where {DT}
    @unpack envsampler, polselect, featurize_x, featurize_op, featurize_pi, = featlearn
    @unpack PredictMat, PolicyMat = featlearn
    @unpack Hmax, N, ctrlrange, isdonefunc, lambdacost = featlearn
    @unpack arg, covW, Amat, covscale = featlearn
    @unpack numfeatures, numopkernel, buffers = featlearn

    if (i == 1) || (mod(i, 50) == 0); @info "Iterations:" i; end;

    #------------------Reset env------------#
    randreset!(envsampler)
    reset!(polselect)
    #-------------------Roll out--------------------#
    elapsed_sample = @elapsed begin
        opt = ControllerIterator((action, state, obs) -> getaction!(action, obs, featurize_pi, PolicyMat.W),
                                 envsampler; T = Hmax, plotiter = 50)
        for _ in opt # runs iterator
        end
        batch = opt.trajectory 
    end


    #-------------------Store data------------------#
    Mmat = reshape(PredictMat.W, numfeatures, numopkernel, numfeatures)
    
    newPhiXdata, newPhiYdata, newOpPhidata, rewdata, Kspectrumdata, Kspectrumdata_est = 
        storedata!(batch, buffers, ctrlrange, 
        featurize_x, featurize_op, PolicyMat, 
        Mmat, isdonefunc, lambdacost, arg)

    if (i == 1) || (mod(i, 1) == 0); @info "rewards" sum(batch.rewards);
        @info "spectrum" Kspectrumdata end;

    sizeofbuf = sum(buffers.sizeofbuf)
    
    
    #------------------Weight matrix----------------#
    @info "Updating Weight Matrix"
    
    preprederr = norm(newPhiYdata - (covW * Amat)' * newOpPhidata)/(norm((covW * Amat)' * newOpPhidata) + 1e-6)
    # matrix inversion lemma / update covariance and compute mean model
    covW .= covW .- covW * newOpPhidata * ((I+newOpPhidata'*covW*newOpPhidata) \ newOpPhidata') * covW
    Amat .+= newOpPhidata * newPhiYdata'
    meanW = covW * Amat
    
    prederr = norm(newPhiYdata - meanW' * newOpPhidata)/(norm(meanW' * newOpPhidata) + 1e-6)
    @info "Mean-model error" preprederr  prederr

    @info "Sampling Matrix"
    # when covscale is set to zero, we use mean model
    if covscale == 0.
        # update weight matrix
        PredictMat.W .= meanW'
    # otherwise, we do Thompson sampling.  Each row is independent.
    else
        TestSampleMat = zeros(size(PredictMat.W,1),size(PredictMat.W,2))
        samplecovmat = Matrix(Hermitian(covW *covscale))
        samplerMat = MvNormal(zeros(size(meanW,1)), samplecovmat)
        for row = 1:size(newPhiXdata,1)
            TestSampleMat[row,:] = meanW[:,row] .+ rand(samplerMat)
        end
       
        # update weight matrix
        PredictMat.W .= TestSampleMat
    end

    @info "Selecting Policy"
    # Pick policy param that returns the best Λ(K) given PredictMat.W 
    Mmat = reshape(PredictMat.W, numfeatures, numopkernel, numfeatures)
    polselect.Mmat .= Mmat
    NextPolMat = reshape(RolloutandSelect(polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))
    @info "norm polpara" norm(NextPolMat-PolicyMat.W)
    PolicyMat.W .= NextPolMat
    @info  PolicyMat.W
    #------------------updating state---------------#
    result = (
              iter = i,
              elapsed_sampled = elapsed_sample,
              traj = batch,
              traj_reward = sum(batch.rewards),
              traj_length = mean(length, batch),
              traj_spectrum = Kspectrumdata,
              traj_spectrum_est = Kspectrumdata_est,
              prederr = prederr
             )

    return result, i + 1
end
