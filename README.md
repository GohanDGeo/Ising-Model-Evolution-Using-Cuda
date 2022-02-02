# Ising-Model-Evolution-Using-Cuda
A project imulating the evolution of a nxn ising model through k steps, using CUDA.
Made for Parallel and Distributed Systems course, ECE AUTH, 2022.

Author: Giorgos Koutroumpis

# How to run
First, download/clone the repository, and navigate to its directory.
## Windows 
Open the CMD and cd to the repos' directory.
Compile using
```
make -f MakeFile
```
and run using
```
ising
```
You can use up to two command line arguments. 
If one argument is provided, it specifies the matrix width n,
while if two arguments are provided, the first specifies n, 
while the second specifies the number of steps k to iterate for.

If no arguments are provided, the variables are defaulted to
`n = 5000, k = 10`.

Eg running for n = 4000, k = 5:
```
make -f MakeFile
ising 4000 5
```

If you don't have make (or a problem occured when using the MakeFile), use this line to compile:
```
nvcc -rdc=true -o ./ising ./src/helpers.cu ./src/kernel.cu ./src/isingEvolution.cu ./src/main.cu
```

## Linux 
Open the CMD and cd to the repos' directory.
Compile using
```
make -f MakeFileLinux
```
and run using
```
ising.out
```
You can use up to two command line arguments. 
If one argument is provided, it specifies the matrix width n,
while if two arguments are provided, the first specifies n, 
while the second specifies the number of steps k to iterate for.

If no arguments are provided, the variables are defaulted to
`n = 5000, k = 10`.

Eg running for n = 4000, k = 5:
```
make -f MakeFileLinux
ising.out 4000 5
```

If you don't have make (or a problem occured when using the MakeFile), use this line to compile:
```
nvcc -rdc=true -o ./ising.out ./src/helpers.cu ./src/kernel.cu ./src/isingEvolution.cu ./src/main.cu
```
