
verify-adam:
	nvcc verify_shmem_adam.cu -Xptxas="-v" -o verify-adam.o -I./include
verify-reg:
	nvcc verify_reg.cu -Xptxas="-v" -o verify-reg.o -I./include
clean:
	rm *.o 
