"""
This file is for SingleIntegrator environment 
"""

using LyceumBase
using Shapes
using Distributions
using Base: @propagate_inbounds

struct SingleIntegrator{D,SP,OP,AP,RP} <: AbstractEnvironment
    state::D
    obs::D
    action::D
    time::D
    ctrlrange::D

    sp::SP
    op::OP
    ap::AP
    rp::RP
    function SingleIntegrator{T}() where T<:AbstractFloat
        nobs = 3
        nstate = 2
        naction = 2

        ctrlrange = zeros(T, 2)
        ctrlrange[1] = -999.;  ctrlrange[2] = 999.
        
        sp = VectorShape(T, nstate)
        op = VectorShape(T, nobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nstate)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{typeof(s), 
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,o,a,zeros(T,1),
                                                        ctrlrange,
                                                        sp,op,ap,rp)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{SingleIntegrator}, n::Integer)
    Tuple(SingleIntegrator{Float64}() for _ in 1:n)
end
SingleIntegrator() = first(tconstruct(SingleIntegrator, 1))

@inline LyceumBase.statespace(env::SingleIntegrator) = env.sp
@inline LyceumBase.obsspace(env::SingleIntegrator) = env.op
@inline LyceumBase.actionspace(env::SingleIntegrator) = env.ap
@inline LyceumBase.rewardspace(env::SingleIntegrator) = env.rp


@inline function LyceumBase.getstate!(state, env::SingleIntegrator) 
    #state .= env.state
    state[1] = env.state[1] # unroll for speed
    state[2] = env.state[2]
end
@inline function LyceumBase.setstate!(env::SingleIntegrator, state) 
    #env.state .= state
    env.state[1] = state[1] # unroll for speed
    env.state[2] = state[2]
end

@inline LyceumBase.getaction!(action, env::SingleIntegrator) = action .= env.action
@inline LyceumBase.setaction!(env::SingleIntegrator, action) = env.action .= action

@propagate_inbounds function LyceumBase.isdone(state, action, obs, env::SingleIntegrator)
    false
end
@propagate_inbounds function LyceumBase.getobs!(obs, env::SingleIntegrator)
    s = env.state
    env.obs[1] = s[1]
    env.obs[3], env.obs[2] = sincos(s[2])
    
    obs .= env.obs
end
@propagate_inbounds function LyceumBase.getreward(s, a, o, env::SingleIntegrator) 
    return 0
end
@propagate_inbounds function LyceumBase.geteval(s, a, o, env::SingleIntegrator)
    return o[1]
end
@propagate_inbounds function LyceumBase.reset!(env::SingleIntegrator)
    env.state[1] = 2.
    env.state[2] = 0.
    getobs!(env.obs, env)
    env.time[1] = 0.0
    env
end
@propagate_inbounds function LyceumBase.randreset!(env::SingleIntegrator)
    env.state[1] = rand(Uniform(0.0, 2.5))
    env.state[2] = rand(Uniform(0, 2π))
    getobs!(env.obs, env)
    env.time[1] = 0.0
    env
end


@propagate_inbounds function LyceumBase.step!(env::SingleIntegrator)
    state, act, ctrlrange = env.state, env.action, env.ctrlrange
    dt = 0.05
    clampaction = clamp.(act, ctrlrange[1], ctrlrange[2])
    ns = copy(state)
    ns[1] += clampaction[1] * dt
    ns[1] = clamp(ns[1], 0, 999)
    ns[2] += clampaction[2] * dt
    
    setstate!(env, ns)
    env.time[1] += timestep(env)
    env
end


@inline LyceumBase.timestep(env::SingleIntegrator) = 1.0
@inline Base.time(env::SingleIntegrator) = env.time[1]

