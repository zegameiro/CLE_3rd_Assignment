canny: src/canny.cu src/image.c src/canny-device.cu
	nvcc -O3 src/canny.cu -o $@

.PHONY: clean
clean:
	@echo "[RM] canny"
	@rm -f canny
	@echo "[RM] *.pgm"
	@rm -f *.pgm
	@echo "[RM] .results"
	@rm -rf results

test: canny
	@echo "[RUN] performance tests"
	@sh performance_tests.sh
	@echo "[OK] Tests finished, results in [results] directory"
