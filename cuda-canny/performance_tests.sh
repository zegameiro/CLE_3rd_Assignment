#!/bin/bash

# Create results directory
mkdir -p results

# CSV header
echo "Image,Width,Height,Sigma,Tmin,Tmax,Host_Time_ms,Device_Time_ms,Speedup" > results/performance_results.csv

# Test parameters
SIGMAS="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5"
TMINS="20 30 40 50 60 70"
TMAXS="80 90 100 110 120 130"
IMAGES="./images/*.pgm"  # Adjust path as needed

echo "Starting performance tests..."

for image in $IMAGES; do
    echo "Testing image: $(basename $image)"
    
    for sigma in $SIGMAS; do
        for tmin in $TMINS; do
            for tmax in $TMAXS; do
                if [ $tmin -lt $tmax ]; then  # Ensure tmin < tmax
                    echo "  Testing sigma=$sigma, tmin=$tmin, tmax=$tmax"
                    
                    # Run the test and capture CSV output
                    # Run program and capture output
                    cmd_output=$(./canny "$image" "results/output_$(basename "$image" .pgm)_s${sigma}_tmin${tmin}_tmax${tmax}.pgm" $tmin $tmax $sigma )
                    
                    # Extract width and height from image dimensions (using ImageMagick's identify)
                    dimensions=$(identify -format "%w,%h" "$image")
                    width=$(echo $dimensions | cut -d',' -f1)
                    height=$(echo $dimensions | cut -d',' -f2)
                    
                    # Extract processing times
                    host_time=$(echo "$cmd_output" | grep "Host processing time" | awk '{print $4}')
                    device_time=$(echo "$cmd_output" | grep "Device processing time" | awk '{print $4}')
                    
                    # Calculate speedup
                    speedup=$(echo "scale=2; $host_time/$device_time" | bc)
                    
                    # Format CSV output
                    host_time_us=$(echo "$host_time" | sed 's/ms/000/')
                    device_time_us=$(echo "$device_time" | sed 's/ms/000/')
                    output="$(basename "$image"),$width,$height,$sigma,$tmin,$tmax,$host_time_us,$device_time_us,$speedup"
                    echo "$output"
                    
                    if [ ! -z "$output" ]; then
                        # Extract the CSV data (remove CSV_RESULT prefix)
                        csv_data=$(echo "$output" | sed 's/CSV_RESULT,//')
                        echo "$csv_data" >> results/performance_results.csv
                    fi
                fi
            done
        done
    done
done

echo "Performance tests completed. Results saved to results/performance_results.csv"