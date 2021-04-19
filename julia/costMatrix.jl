using CSV, DataFrames, DelimitedFiles
using SparseArrays, NPZ
using LinearAlgebra
using ArgParse

# julia costMatrix.jl --dataset ml-20m --datasize small --model vae

s = ArgParseSettings()
@add_arg_table s begin
    "--dataset"
        help = "ml-20m, netflix"
        arg_type = String
    "--datasize"
        help = "large, small"
        arg_type = String
    "--model"
        help = "vae, mf (only used for small datasize)"
        arg_type = String
end

args = parse_args(s)
dataset = args["dataset"]
datasize = args["datasize"]
model = args["model"]
#@assert dataset in datasets "$(dataset) not found in $datasets"

datadir = "/home/xx/data/$(dataset)/$(datasize)"
if datasize == "small"
    datadir = joinpath(datadir, model)
end
shrink = datasize == "small" ? (dataset == "netflix" ? 1.0f0 : 5.0f0) : 50.0f0

df = CSV.File(joinpath(datadir, "train.csv")) |> DataFrame
## check the true training data
if "istrain" in names(df)
    df = df[df.istrain .>= 1.0, :]
end

I = df.uid .+ 1 ## user
J = df.sid .+ 1 ## item
V = ones(Float32, size(df, 1))
A = sparse(I, J, V)
println("#Users: $(size(A, 1)), #Items: $(size(A, 2))")

n = size(A, 2)
## L2 norm of each item (column)
itemLengths = mapslices(x -> norm(x), A, dims=1) |> Matrix |> vec

## Memory expensive way
# S = A' * A
# S ./= itemLengths * itemLengths' .+ shrink - LinearAlgebra.I * shrink
# S = convert(Matrix{Float16}, Matrix(S))

## Memory efficient way
S = A' * A |> Matrix
for i = 1:n
    S[:, i] ./= itemLengths * itemLengths[i] .+ shrink
    S[i, i] = 1.0
end
S = convert(Matrix{Float16}, S)
#S[S .< 0.5] .= Float16(0.0)

cmfile = "cm.npy"
println("Saving cost matrix into $(joinpath(datadir, cmfile))")
npzwrite(joinpath(datadir, cmfile), Float16(1.0) .- S)
