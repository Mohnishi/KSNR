using Revise
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays, ElasticArrays
using Distances

using LaTeXStrings
using Plots.PlotMeasures
using Formatting

include("mujoco_models/walker2d.jl")

# # make directories to store log files
_ckmkdir(f) = isdir(f) == false && mkdir(f)
_ckmkdir("log")
_ckmkdir("plot")

# #--------------- execute ---------------------# 
if executescripts
    seednum = 100; dfile_num = "1"
    include("scripts/walker2d_sim.jl")
    seednum = 200; dfile_num = "2"
    include("scripts/walker2d_sim.jl")
    seednum = 300; dfile_num = "3"
    include("scripts/walker2d_sim.jl")
    seednum = 400; dfile_num = "4"
    include("scripts/walker2d_sim.jl")
end
#--------------- compute/plot the results ---------------# 
d3 = []
Koopman_gt = []; Koopman_sp = [];
Lcost_gt = []; Lcost_sp = []; cumulative_gt = [];
cumulative_sp = []; opt = []; opt_sp = [];

for i in 1:4
    push!(d3, read("log/data3_"*string(i)*".jlso", JLSOFile))
    push!(Koopman_gt, d3[i][:Koopman_gt])
    push!(Koopman_sp, d3[i][:Koopman_sp])
    push!(Lcost_gt, d3[i][:Lcost_gt])
    push!(Lcost_sp, d3[i][:Lcost_sp])
    push!(cumulative_gt, d3[i][:cumulative_gt])
    push!(cumulative_sp, d3[i][:cumulative_sp])
    push!(opt, d3[i][:opt])
    push!(opt_sp, d3[i][:opt_sp])
end

mean_Lcost_gt = mean(Lcost_gt); mean_Lcost_sp = mean(Lcost_sp);
mean_cumulative_gt = mean(cumulative_gt); mean_cumulative_sp = mean(cumulative_sp);
std_Lcost_gt = std(Lcost_gt); std_Lcost_sp = std(Lcost_sp);
std_cumulative_gt = std(cumulative_gt); std_cumulative_sp = std(cumulative_sp);


println("The cumulative reward when NOT using spectrum cost->")
println("mean: $mean_cumulative_gt, std: $std_cumulative_gt")
println("The spectrum cost when NOT using spectrum cost->")
println("mean: $mean_Lcost_gt, std: $std_Lcost_gt")
println("The cumulative reward when using spectrum cost->")
println("mean: $mean_cumulative_sp, std: $std_cumulative_sp")
println("The spectrum cost when using spectrum cost->")
println("mean: $mean_Lcost_sp, std: $std_Lcost_sp")


titles = ["Joint trajectories (no spectrum cost)" "Joint trajectories with spectrum cost"]
titles2 = ["Eigenspectrum" "" ""]
#--------------##--------------##--------------#
println("Saving joint trajectory.")
Odata = Matrix(opt[1].trajectory.observations)
Odata_sp = Matrix(opt_sp[1].trajectory.observations)
eigs = Vector(sort(abs.(eigvals(Koopman_gt[1])), rev=true))
eigs_sp = Vector(sort(abs.(eigvals(Koopman_sp[1])), rev=true))

#--------------#
pl1 = plot(1:size(Odata,2), Odata[3:8,:]', grid=true, 
xlabel="timesteps", ylabel="values", xlim=(0,size(Odata,2)),
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 6,framestyle=:box, size = (1600, 1000),
bottom_margin=30mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
pl2 = plot(1:size(Odata_sp,2), Odata_sp[3:8,:]', grid=true, 
xlabel="timesteps", ylabel="values", xlim=(0,size(Odata_sp,2)),
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 6,framestyle=:box, size = (1600, 1000),
bottom_margin=30mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
ple = plot(1:length(eigs), eigs, grid=true, 
xlabel="index", ylabel="absolute values", xlim=(1,length(eigs)),
guidefontsize=30, tickfontsize=30, foreground_color_border="gray", 
legend=false,linewidth = 4, linecolor="black", linestyle=:dot, framestyle=:box, size = (1600, 1000),
bottom_margin=30mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
plot!(1:length(eigs_sp), eigs_sp,  linewidth = 4,
linecolor="black", linestyle=:solid,legend=false)
#--------------#
plegend1 = plot(1:1,linewidth=4,
linestyle=:dot,linecolor="black",
legend=:left,label = "no spectrum cost",
framestyle = :none,legendfontsize=30,
foreground_color_legend = nothing)

plegend2 = plot(1:1,linewidth=4,
linestyle=:solid,linecolor="black",
legend=:left,label = "with spectrum cost",
framestyle = :none,legendfontsize=30,
foreground_color_legend = nothing)

#--------------#
plot1 = plot(pl1,pl2,
layout=@layout([a b]), size = (3200, 1000),
title=titles, titlefontsize=40)
savefig(plot1, "plot/plot_walker.png")

#--------------#
plot1e = plot(ple,plegend1,plegend2,
layout=@layout([a;b{0.1h} c{0.1h}]), size = (1600, 1100),
title=titles2, titlefontsize=30)
savefig(plot1e, "plot/plot_walker_eig.png")