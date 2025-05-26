
canny: src/canny.cu src/image.c
	nvcc -O3 src/canny.cu -o $@

.PHONY: clean
clean:
	@echo "[RM] canny"
	@rm -f canny
	@echo "[RM] *.pgm"
	@rm -f *.pgm

canny-device: src/canny-device.cu src/image.c
	nvcc -03 src/canny-device.cu -o $@
