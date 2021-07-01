"""
This file is for creating RFF structure and for fast evaluations at given inputs.
"""

using Random, SparseArrays

# A set of random fourier functions
struct RandomFourierFunctions{T<:AbstractFloat}
	directions::Matrix{T} # random direction
    offsets::Vector{T} # random shift in [0,2*pi]
    sigma::T

    function RandomFourierFunctions{T}(bandwidth, input, num_functions) where T<:AbstractFloat
        w = randn(T, num_functions, input) / T(bandwidth)
        # draw random offsets from a uniform distribution in [-pi,pi]
        b = rand(T, (num_functions,)) * T(2pi) .- T(pi)
        new{T}(w, b, T(bandwidth))
    end
end

function (rff::RandomFourierFunctions)(x::AbstractArray)
    W, b = rff.directions, rff.offsets
	nfeat = size(W,1)
    sD = eltype(b)(sqrt(2/nfeat))
    sD .* cos.(W*x .+ b)
end
(rff::RandomFourierFunctions{T})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}} = invoke(rff, Tuple{AbstractArray}, x)
(rff::RandomFourierFunctions{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = rff(T.(x))

function (rff::RandomFourierFunctions{T})(z::AbstractArray{T, N},
                                          x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
    nfeat = size(rff.directions,1)
    sD = T(sqrt(2/nfeat))
   
    W, b = rff.directions, rff.offsets
    mul!(z, W, x)
    z .+= b
    z .= sD .* cos.(z)
    z
end
##############################
##############################

struct RandomFourierFunctionsAdditive{T<:AbstractFloat}
	directions::Matrix{T} # random direction
    offsets::Vector{T} # random shift in [0,2*pi]
    sigma::T

    function RandomFourierFunctionsAdditive{T}(bandwidth, input, num_functions) where T<:AbstractFloat
        mod(num_functions,input) == 0 || throw(ArgumentError("num_functions must be a multiply of input"))
        numfunc_input = Int(num_functions/input)
        P = randn(T, input, input)
        w = Matrix(blockdiag([sprandn(T, numfunc_input, 1, 1.) / T(bandwidth) for _ in 1:input]...)) * P
       
        b = rand(T, (num_functions,)) * T(2pi) .- T(pi)
        new{T}(w, b, T(bandwidth))
    end
end

function (rff::RandomFourierFunctionsAdditive)(x::AbstractArray)
    W, b = rff.directions, rff.offsets
	nfeat = size(W,1)
    sD = eltype(b)(sqrt(2/nfeat))
    sD .* cos.(W*x .+ b)
end
(rff::RandomFourierFunctionsAdditive{T})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}} = invoke(rff, Tuple{AbstractArray}, x)
(rff::RandomFourierFunctionsAdditive{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = rff(T.(x))

function (rff::RandomFourierFunctionsAdditive{T})(z::AbstractArray{T, N},
                                          x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
    nfeat = size(rff.directions,1)
    sD = T(sqrt(2/nfeat))
   
    W, b = rff.directions, rff.offsets
    mul!(z, W, x)
    z .+= b
    z .= sD .* cos.(z)
    z
end


##############################
##############################

struct LinearFeature{T<:AbstractFloat}
	dim::Int # random direction

    function LinearFeature{T}(input) where T<:AbstractFloat
        dim = input
        new{T}(dim)
    end
end

function (LinearFeature::LinearFeature)(x::AbstractArray)
    if size(x,2)==1
        return [x; 1.]
    end
    return [x; ones(1,size(x,2))]
end
(LinearFeature::LinearFeature{T})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}} = invoke(LinearFeature, Tuple{AbstractArray}, x)
(LinearFeature::LinearFeature{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = LinearFeature(T.(x))

function (LinearFeature::LinearFeature{T})(z::AbstractArray{T, N},
                                          x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
    if size(x,2)==1
        y = [x; 1]
    else
        y = [x; ones(1,size(x,2))]
    end
    z .= y
    z
end

##############################
##############################
struct LinearRandom{T<:AbstractFloat}
	directions::Matrix{T} # random direction
    offsets::Vector{T} # random shift in [0,2*pi]
    sigma::T

    function LinearRandom{T}(bandwidth, input, num_functions) where T<:AbstractFloat
        w = randn(T, num_functions-input, input) / T(bandwidth)
        # draw random offsets from a uniform distribution in [-pi,pi]
        b = rand(T, (num_functions-input,)) * T(2pi) .- T(pi)
        new{T}(w, b, T(bandwidth))
    end
end

function (rff::LinearRandom)(x::AbstractArray)
    W, b = rff.directions, rff.offsets
	nfeat = size(W,1)
    sD = eltype(b)(sqrt(2/nfeat))
    output = sD .* cos.(W*x .+ b)
    [x;output]
end
(rff::LinearRandom{T})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}} = invoke(rff, Tuple{AbstractArray}, x)
(rff::LinearRandom{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = rff(T.(x))

function (rff::LinearRandom{T})(z::AbstractArray{T, N},
                                          x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
    nfeat = size(rff.directions,1)
    sD = T(sqrt(2/nfeat))
   
    W, b = rff.directions, rff.offsets
    mul!(z[length(x)+1:end], W, x)
    z[length(x)+1:end] .+= b
    z[length(x)+1:end] .= sD .* cos.(z[length(x)+1:end])
    z[1:length(x)] .= x
    z
end



