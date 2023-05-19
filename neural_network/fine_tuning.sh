#!/bin/bash

# ITERATIONS

start=100    # Starting value
end=7000     # Ending value
step=100     # Step size

# Generate the sequence and store it in an array
max_iterations=()
for ((i=start; i<=end; i+=step)); do
  max_iterations+=($i)
done

# ALPHA

start=0.1    # Starting value
end=50.0       # Ending value
step=0.1     # Step size

# Generate the sequence and store it in an array
alpha=()
for ((i=start; i<=end; i+=step)); do
  alpha+=($i)
done

# BETA

beta=("${alpha[@]}")

# THRESHOLD

start=0.1    # Starting value
end=3.0       # Ending value
step=0.1     # Step size

# Generate the sequence and store it in an array
threshold=()
for ((i=start; i<=end; i+=step)); do
  threshold+=($i)
done

# HIDDEN SIZE

hidden_size=(310 300 250 200 150 100 50 25 10)

# OUTPUT SIZE

output_size=(310 300 250 200 150 100 50 25 10)

# HIDDEN SIZE + OUTPUT SIZE

combinations=()
for iter in "${max_iterations[@]}"; do
    for a in "${alpha[@]}"; do
        for b in "${beta[@]}"; do
            for th in "${threshold[@]}"; do
                for hidden in "${hidden_size[@]}"; do
                    for output in "${output_size[@]}"; do
                        if ((output <= hidden)); then
                            python3 main.py --max_iterations $iter --alpha $a --beta $b --threshold $th --hidden_size $hidden --output_size $output
                        fi
                    done
                done
            done
        done
    done
done
