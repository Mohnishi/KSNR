## Readme

This code is meant to reproduce the results found in **Koopman Spectrum Nonlinear Regulator and Provably Efficient Online Learning** by Motoya Ohnishi, Isao Ishikawa, Kendall Lowrey, Masahiro Ikeda, Sham Kakade, and Yoshinobu Kawahara.
Some experiments require external licence, namely, [MuJoCo](http://www.mujoco.org/).
Refer to [Lyceum MuJoCo](https://github.com/Lyceum/MuJoCo.jl) for instructions
on how to use MuJoCo under Lyceum platform.

## Setup & Install

This code has been tested on Ubuntu 18.04, but should also work on different platforms (MacOS, Windows, FreeBSD) if the instructions are adapted.

The process to bring up this repo is as follows:
1. Download and install [Julia](https://julialang.org/)
2. Navigate to project and instantiate 
3. Run

The following is an example of installing Julia for Ubuntu 18.04.
```bash
cd ~/Downloads
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
tar xvf julia-1.5.3-linux-x86_64.tar.gz

# the following exports can be added to your bashrc.
export JULIA_BINDIR=~/Downloads/julia-1.5.3/bin
export PATH=$JULIA_BINDIR:$PATH
export JULIA_NUM_THREADS=12

cd $directory_you_extracted_code
julia
```

Once you start Julia, regardless of platform, the following instructions may proceed:
```julia
julia> ]
(@v1.5) pkg> registry add https://github.com/Lyceum/LyceumRegistry     # add Lyceum registry
(@v1.5) pkg> activate .                        # activates this project
(KSNR) pkg> instantiate   # the built in package manager downloads, installs dependences
(KSNR) pkg> ctrl-c

julia> executescripts = false                      # to use our data to plot;   executescripts = true  is for running the algorithms
julia> include("main_1.jl")                      # to run the first experiment and/or plot the data  (limit-cycle experiment);  include("main_2.jl"), include("main_3.jl"), include("main_4.jl") does each experiment.
```

## Notes

The results in the paper were generated with **Julia 1.5.3**, with **12 Julia threads**. This is critical to reproducibility, but not necessary for running the included algorithm; one should adapt these settings to their compute.

Also, **one may need to restart Julia to run experiments sequentially**.  To exit julia, do
```julia
julia> exit()
```
Next time you start Julia, you do not need to do instantiate but only activate.

## Code Structure

```bash
.
├── log                    # Data store
│   ├── data1.jlso
│   ├── data2.jlso
│   └── ...
├── main_1.jl
├── main_2.jl
├── main_3.jl
├── main_4.jl
├── Manifest.toml          # Julia Manifest file for all dependencies
├── models           # Analytical dynamical system models
│   ├── cartpole.jl
│   └── singleint.jl
├── mujoco_models                    # MuJoCo models (require MuJoCo)
│   ├── cartpole_stable.jl
│   ├── cartpole_stable.xml
│   ├── walker2d.jl
│   ├── walker2d.xml
│   ├── reward.jl
│   └── common
│       ├── materials.xml
│       ├── skybox.xml
│       └── visual.xml
├── planner           # Heuristic planner algorithms
│   ├── MPPIClamp.jl
│   ├── PolicySelect.jl
│   ├── PolicySelect-GT.jl
│   └── PolicySelect-SP.jl
├── plot
├── Project.toml           # Julia Project file for top level dependencies
├── README.md              # This file
├── scripts                # Environment Hyper-Parameters and configuration/ Running
│   ├── cartpole_sim.jl
│   ├── learning.jl
│   ├── learning_gt.jl
│   ├── learning_run.jl
│   ├── singleint_sim.jl
│   └── walker2d_sim.jl
└── utils                  # Algorithm and support code
    ├── algorithm.jl
    ├── learned_env.jl
    ├── rff.jl
    └── weightmat.jl
```

Note walker2d is from [OpenAI Gym](https://github.com/openai/gym) and materials, skybox, visual, and cartpole are from [DeepMind Control Suite](https://github.com/deepmind/dm_control)
 
## Code Maintenance

The codes are maintained by the authors of **Koopman Spectrum Nonlinear Regulator and Provably Efficient Online Learning** ([arXiv](https://arxiv.org/abs/2106.15775)).
The project page can be found [here](https://sites.google.com/view/ksnr-dynamics/)
