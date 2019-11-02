export OMP_NUM_THREADS=1
for batch_size in 20 30 40 50 100 200
do
     python run.py --seed=130 --algorithm=OEI --function=eggholder --batch_size=$batch_size --initial_size=10 --iterations=1 --noise=1e-6 --hessian=0 --nl_solver=bfgs
done