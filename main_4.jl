using Revise
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays, ElasticArrays
using Distances

using LaTeXStrings
using Plots.PlotMeasures
using Formatting
include("models/cartpole.jl")
# # make directories to store log files
_ckmkdir(f) = isdir(f) == false && mkdir(f)
_ckmkdir("log")
_ckmkdir("plot")

# #--------------- execute ---------------------# 
if executescripts
    include("scripts/learning.jl")
    seednum = 100; dfile_num = "1"
    include("scripts/learning_gt.jl")
    include("scripts/learning_run.jl")
    seednum = 200; dfile_num = "2"
    include("scripts/learning_gt.jl")
    include("scripts/learning_run.jl")
    seednum = 300; dfile_num = "3"
    include("scripts/learning_gt.jl")
    include("scripts/learning_run.jl")
    seednum = 400; dfile_num = "4"
    include("scripts/learning_gt.jl")
    include("scripts/learning_run.jl")
end
#--------------- compute/plot the results ---------------# 
d4 = []; d4gt = [];
cemKoopman_gt = []; cemLcost_gt = []; cemopt_gt = []; cemcumulative_gt = [];
cemKoopman_sp = []; cemLcost_sp = []; cemopt_sp = []; cemcumulative_sp = [];
traj_spectrum = []; traj_spectrum_est = []; 
for i in 1:4
    push!(d4, read("log/data4_"*string(i)*".jlso", JLSOFile))
    push!(d4gt, read("log/data4gt_"*string(i)*".jlso", JLSOFile))
    push!(cemKoopman_gt, d4gt[i][:Koopman_gt])
    push!(cemKoopman_sp, d4gt[i][:Koopman_sp])
    push!(cemLcost_gt, d4gt[i][:Lcost_gt])
    push!(cemLcost_sp, d4gt[i][:Lcost_sp])
    push!(cemopt_gt, d4gt[i][:opt])
    push!(cemopt_sp, d4gt[i][:opt_sp])
    push!(cemcumulative_gt, d4gt[i][:cumulative_gt])
    push!(cemcumulative_sp, d4gt[i][:cumulative_sp])
    push!(traj_spectrum, d4[i][:logs][:algstate][:traj_spectrum])
    push!(traj_spectrum_est, d4[i][:logs][:algstate][:traj_spectrum_est])
end
traj = d4[1][:traj]
#--------------##--------------##--------------#
mean_cemLcost_gt = mean(cemLcost_gt); mean_cemLcost_sp = mean(cemLcost_sp);
mean_cemcumulative_gt = mean(cemcumulative_gt);
mean_cemcumulative_sp = mean(cemcumulative_sp);
std_cemLcost_gt = std(cemLcost_gt); std_cemLcost_sp = std(cemLcost_sp);
std_cemcumulative_gt = std(cemcumulative_gt); 
std_cemcumulative_sp = std(cemcumulative_sp);


println("The cem cumulative reward when NOT using spectrum cost->")
println("mean: $mean_cemcumulative_gt, std: $std_cemcumulative_gt")
println("The cem spectrum cost when NOT using spectrum cost->")
println("mean: $mean_cemLcost_gt, std: $std_cemLcost_gt")
println("The cem cumulative reward when using spectrum cost->")
println("mean: $mean_cemcumulative_sp, std: $std_cemcumulative_sp")
println("The cem spectrum cost when using spectrum cost->")
println("mean: $mean_cemLcost_sp, std: $std_cemLcost_sp")
#--------------##--------------##--------------#

titles = ["Joint trajectories (no spectrum cost)" "Joint trajectories with spectrum cost"]
titles2 = ["Eigenspectrum" "" ""]
#--------------##--------------##--------------#
println("Saving trajectory.")
cemobs_gt = Vector(cemopt_gt[1].trajectory.observations[1,:])
cemobs_sp = Vector(cemopt_sp[1].trajectory.observations[1,:])
obs_learn = Vector(traj.observations[1,:])
#--------------#
pl1 = plot(0:length(cemobs_gt)-1, [cemobs_gt cemobs_sp obs_learn],
grid=true, xlabel="timesteps", ylabel="cart position", 
xlim=(0,length(cemobs_gt)),
guidefontsize=45, tickfontsize=35, linecolor = ["blue" "red" "red"],
linestyle=[:dot :dot :solid], foreground_color_border="gray", 
legend=false,linewidth = 7,framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=30mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true, title="Cart position trajectories",
titlefontsize = 45)
#--------------#
lenr = length(traj_spectrum[1])
mean_traj_spectrum = Vector(mean(traj_spectrum))
std_traj_spectrum = Vector((traj_spectrum[1].^2+
traj_spectrum[2].^2+traj_spectrum[3].^2+traj_spectrum[4].^2)/4)
mean_traj_spectrum_est = Vector(mean(traj_spectrum_est))
std_traj_spectrum_est = Vector((traj_spectrum_est[1].^2+
traj_spectrum_est[2].^2+traj_spectrum_est[3].^2+traj_spectrum_est[4].^2)/4)
mean_move_spectrum = zeros(lenr)
std_move_spectrum = zeros(lenr)
mean_move_spectrum_est = zeros(lenr)
std_move_spectrum_est = zeros(lenr)
    for i in 1:lenr
        numbermove = 0.
        for j in max(1,(i-4+1)):i
            numbermove += 1.
            mean_move_spectrum[i] += mean_traj_spectrum[j];
            std_move_spectrum[i] += std_traj_spectrum[j]; 
            mean_move_spectrum_est[i] += mean_traj_spectrum_est[j];
            std_move_spectrum_est[i] += std_traj_spectrum_est[j]; 
        end
        mean_move_spectrum[i] /= numbermove
        std_move_spectrum[i] /= numbermove
        mean_move_spectrum_est[i] /= numbermove
        std_move_spectrum_est[i] /= numbermove
end
std_move_spectrum .= sqrt.(std_move_spectrum - mean_move_spectrum.^2)
std_move_spectrum_est .= sqrt.(std_move_spectrum_est - mean_move_spectrum_est.^2)

plotspec1 = plot(1:length(mean_move_spectrum),mean_move_spectrum,grid=true,
    ribbon=std_move_spectrum,fillalpha=.15, 
    xlabel="episodes", ylabel="spectrum cost",xlim=(1,length(mean_move_spectrum)),
    xticks = 1:4:20,
    guidefontsize=45, tickfontsize=35, foreground_color_border="gray", 
    linewidth = 7,framestyle=:box, size = (1200, 1000),
    linecolor = "green",
    bottom_margin=50mm,left_margin=25mm,right_margin=30mm,top_margin=13mm,fillcolor="green",legend=false,
    yformatter=y->format( y, precision= 2),xformatter=x->format( x, precision= 0),
    title="spectrum cost curve",titlefontsize = 45)

plotspec2 = plot(1:length(mean_move_spectrum_est),
    mean_move_spectrum_est,grid=true,
    ribbon=std_move_spectrum_est,fillalpha=.15, 
    xlabel="episodes", ylabel="spectrum cost",xlim=(1,length(mean_move_spectrum_est)),
    xticks = 1:4:20,
    guidefontsize=45, tickfontsize=35, foreground_color_border="gray", 
    linewidth = 7,framestyle=:box, size = (1200, 1000),
    linecolor = "green",
    bottom_margin=50mm,left_margin=13mm,right_margin=32mm,top_margin=13mm,fillcolor="green",legend=false,
    yformatter=y->format( y, precision= 2),xformatter=x->format( x, precision= 0),
    title="estimated spectrum cost curve",titlefontsize = 45)

plegend1 = plot(1:1,linewidth=4,
    linestyle=:dot,linecolor="blue", 
    legend=:left,label = "no spectrum cost (CEM)",
    framestyle = :none,legendfontsize=40,
    foreground_color_legend = nothing)

plegend2 = plot(1:1,linewidth=4,
    linestyle=:dot,linecolor="red", 
    legend=:left,label = "with spectrum cost (CEM)",
    framestyle = :none,legendfontsize=40,
    foreground_color_legend = nothing)

plegend3 = plot(1:1,linewidth=4,
    linestyle=:solid,linecolor="red", 
    legend=:left,label = "with spectrum cost (learn)",
    framestyle = :none,legendfontsize=40,
    foreground_color_legend = nothing)

pl = plot(pl1,plotspec1,plotspec2,plegend1,plegend2,plegend3,
    layout=@layout([a b c; d{0.1h} e{0.1h} f{0.1h}]), size = (3600, 1100))
    
savefig(pl, "plot/plot_learning.png")
