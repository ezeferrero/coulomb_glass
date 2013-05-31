
struct timespec diff(timespec start, timespec end)
{
        timespec temp;
       	if ((end.tv_nsec-start.tv_nsec)<0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } else {
                temp.tv_sec = end.tv_sec-start.tv_sec;
                temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        }
        return temp;
}


// un timer para procesos en la CPU
// USO: cpu_timer T; T.cpu_tic();...calculo...;T.cpu_tac();T.print_cpu_elapsed();
class cpu_timer{
	private:
        struct timespec time1, time2;
        float mean_elapsed_time;
        float dev_elapsed_time;
        float elapsed_time;
        unsigned long count;

        public:
        cpu_timer(){
        	reset();
        }
        void reset(){
            mean_elapsed_time=dev_elapsed_time=0.0;
            count=0;
        }
        void tic(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        }
        void tac(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
                elapsed_time=cpu_elapsed();
                mean_elapsed_time+=elapsed_time;
                dev_elapsed_time+=elapsed_time*elapsed_time;
                count++;
        }
        void print_stat(){
            	float av = mean_elapsed_time/count;
            	float dev= sqrt(dev_elapsed_time/count - av*av)/sqrt(count);
            	if(count==1)
            		printf("CPU Time elapsed %.8f  ms\n",av);
            	else
            		printf("CPU Time elapsed %.8f +- %.8f ms (%ld samples)\n",av,dev,count);
        }
        float print(){
            	printf("CPU Time elapsed %.8f  ms\n",elapsed_time);
		return elapsed_time;
        }
        float cpu_elapsed(){
            return (float)diff(time1,time2).tv_sec*1000 + (float)diff(time1,time2).tv_nsec*0.000001;
        }
       ~cpu_timer(){}
};
