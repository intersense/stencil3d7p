jacobi7origin:
	gcc testorigin.c -o jacobi7.o -I./include

jacobi7-cuda:
	nvcc jacobi7-cuda-naive.cu -o jacobi7-cuda-naive.o -I./include

verify-shmem:
	nvcc verify_shmem.cu -o verify-shmem.o -I./include
verify-shmem2:
	nvcc verify_shmem2.cu -o verify-shmem2.o -I./include
verify-glmem:
	nvcc verify_glmem.cu -o verify-glmem.o -I./include
verify-glmem-no-timer:
	nvcc verify_glmem_no_timer.cu -o verify-glmem.o -I./include
verify-glmem-l1-cache:
	nvcc verify_glmem_no_timer.cu -Xptas="-dlcm=ca" -o verify-glmem.o -I./include
verify-shmem_3d:
	nvcc verify_shmem_3d.cu -o verify-shmem_3d.o -I./include
verify-adam:
	nvcc verify_shmem_adam.cu -Xptxas="-v" -o verify-adam.o -I./include
verify-adam-reg:
	#nvcc verify_shmem_adam_reg.cu -Xptxas="-v" -maxrregcount 16 -o verify-adam-reg.o -I./include
	nvcc verify_shmem_adam_reg.cu -Xptxas="-v" -o verify-adam-reg.o -I./include
verify-overlap:
	nvcc verify_overlap.cu -o verify-overlap.o -I./include
verify-overlap-adam:
	nvcc verify_adam_overlap.cu -o verify-overlap-adam.o -I./include
verify-overlap0:
	nvcc verify_overlap0.cu -o verify-overlap0.o -I./include
verify-temporal:
	nvcc verify_temporal.cu -o verify-temporal.o -I./include
verify-temporal-adam:
	nvcc verify_adam_temporal.cu -o verify-adam-temporal.o -I./include
verify-35d:
	nvcc verify_35d.cu -o verify-35d.o -I./include
verify-35d-halo-test:
	nvcc verify_35d_halo_test.cu -o verify-35d-halo-test.o -I./include
verify-test:
	nvcc verify_shmem_test.cu -o jacobi3d_7p_shmem_test.o -I./include
clean:
	rm *.o 
