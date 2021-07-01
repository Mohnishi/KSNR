"""
This file is for creating a virtual environment reflecting the learned model.
For our work, we use feature -> feature as a prediction and use feature for reward computation etc.
In our case, feature includes obs too
"""

using LyceumBase
using Shapes
using Base: @propagate_inbounds

# we assume the data stored in state, obs, action are vectors of some float type
struct LearnedEnv{D,E,K,P,R,SP,OP,AP,RP} <: AbstractEnvironment
    state::D
    featobs::D
    firstobs::D
    action::D
    time::D
    mjenv::E 
    KoopmanMat::K
    phi::P
    rewfunc::R
    nextfeatobs::D
    sp::SP
    op::OP
    ap::AP
    rp::RP
    function LearnedEnv{T}(KoopmanMat, phi, rewfunc, mjenv) where T<:AbstractFloat
        nfeatobs = size(KoopmanMat, 2)
        nobs = length(obsspace(mjenv))
        naction = length(actionspace(mjenv))
        sp = VectorShape(T, nfeatobs)
        op = VectorShape(T, nfeatobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nfeatobs)
        fo = zeros(T, nfeatobs)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{typeof(s), typeof(mjenv), typeof(KoopmanMat), typeof(phi), typeof(rewfunc),
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,fo,o,a,zeros(T,1),
                                                        mjenv,
                                                        copy(KoopmanMat), 
                                                        phi,
                                                        rewfunc,
                                                        zeros(T, nfeatobs),
                                                        sp,op,ap,rp)
        LyceumBase.reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{LearnedEnv}, KoopmanMat, phi, rewfunc, mjenvs, n)
    @assert length(mjenvs) >= n
    Tuple(LearnedEnv{Float64}(KoopmanMat, phi, rewfunc, mjenv) for mjenv in mjenvs)
end
LearnedEnv(KoopMat, phi, rewfunc, mjenv) = first(tconstruct(LearnedEnv, KoopMat, phi, rewfunc, [mjenv], 1))



@inline LyceumBase.statespace(env::LearnedEnv) = env.sp
@inline LyceumBase.obsspace(env::LearnedEnv) = env.op
@inline LyceumBase.actionspace(env::LearnedEnv) = env.ap
@inline LyceumBase.rewardspace(env::LearnedEnv) = env.rp

@inline function LyceumBase.getstate!(state, env::LearnedEnv) 
    state .= 0.0 
    copyto!(state, env.state)
end
@inline function LyceumBase.setstate!(env::LearnedEnv, state) 
    if length(env.state) >= length(state)
        env.state .= state
    else
        nstate = length(env.state) 
        for i=1:nstate
            env.state[i] = state[i]
        end
    end
end

@inline LyceumBase.getaction!(action, env::LearnedEnv) = action .= env.action
@inline LyceumBase.setaction!(env::LearnedEnv, action) = env.action .= action


@propagate_inbounds function LyceumBase.getobs!(obs, env::LearnedEnv)
    copyto!(env.featobs, env.state)
    obs .= env.featobs
end
@propagate_inbounds function LyceumBase.getreward(feato, env::LearnedEnv)
    env.rewfunc(feato)
end

@propagate_inbounds function LyceumBase.reset!(env::LearnedEnv)
    reset!(env.mjenv)
    getobs!(env.firstobs, env.mjenv)
    env.phi(env.featobs, env.firstobs)
    nstate = length(env.state) 
    copyto!(env.state, 1:nstate, env.featobs, 1:nstate)
    env.time[1] = 0.0
    env
end
@propagate_inbounds function LyceumBase.randreset!(env::LearnedEnv)
    randreset!(env.mjenv)
    getobs!(env.firstobs, env.mjenv)
    env.phi(env.featobs, env.firstobs)
    nstate = length(env.state) 
    copyto!(env.state, 1:nstate, env.featobs, 1:nstate)
    env.time[1] = 0.0
    env
end

@propagate_inbounds function LyceumBase.step!(env::LearnedEnv)
    # state = KoopmanMat * phi(obs)
    featobs, act = env.featobs, env.action
    mul!(env.nextfeatobs, env.KoopmanMat, featobs)
    env.state .= env.nextfeatobs
    setstate!(env, env.state)
    env.time[1] += timestep(env)
    env
end



@inline LyceumBase.timestep(env::LearnedEnv) = timestep(env.mjenv)
@inline Base.time(env::LearnedEnv) = env.time[1]
