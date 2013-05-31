#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <string>
#include <vector>

# if  (defined(__MACH__) && defined(__APPLE__))
#   define _MAC
# elif (defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(_WIN64))
#   define _WINDOWS
#   ifndef WIN32_LEAN_AND_MEAN
#     define WIN32_LEAN_AND_MEAN
#   endif
#endif

# if defined(_MAC)
#    include <mach/mach_time.h>
# elif defined(_WINDOWS)
#    include <windows.h>
# else
#    include <time.h>
# endif


#if defined(_MAC)
  typedef uint64_t timer_t;
  typedef double   timer_c;

#elif defined(_WINDOWS)
  typedef LONGLONG      timer_t;
  typedef LARGE_INTEGER timer_c;

#else
  typedef double   timer_t;
  typedef timespec timer_c;
#endif

  //==============================================================================
  // cpu_timer
  // A quick class to do benchmarking.
  // Example: cpu_timer t;  t.tic();  SomeSlowOp(); t.tac("Some Message");

  class cpu_timer {
  public:
    cpu_timer();

    inline void tic();
    inline void tac();
    inline void tac(const std::string &msg);

    void print(const std::string &msg);
    float print();
    void reset();
    double getTime();

  private:
    timer_t start;
    double duration;
    timer_c ts;
    double conv_factor;
    double elapsed_time;
  };



  cpu_timer::cpu_timer() {

#if defined(_MAC)
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);

    conv_factor = (static_cast<double>(info.numer))/
                  (static_cast<double>(info.denom));
    conv_factor = conv_factor*1.0e-9;

#elif defined(_WINDOWS)
    timer_c freq;
    QueryPerformanceFrequency(&freq);
    conv_factor = 1.0/(static_cast<double>freq.QuadPart);

#else
    conv_factor = 1.0;
#endif

    reset();
  }

  inline void cpu_timer::tic() {

#if defined(_MAC)
    start = mach_absolute_time();

#elif defined(_WINDOWS)
    QueryPerformanceCounter(&ts);
    start = ts.QuadPart;

#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    start = static_cast<double>(ts.tv_sec) + 1.0e-9 *
            static_cast<double>(ts.tv_nsec);

#endif
  }

  inline void cpu_timer::tac() {
#if defined(_MAC)
    duration =  static_cast<double>(mach_absolute_time() - start);

#elif defined(_WINDOWS)
    QueryPerformanceCounter(&qpc_t);
    duration = static_cast<double>(qpc_t.QuadPart - start);

#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    duration = (static_cast<double>(ts.tv_sec) + 1.0e-9 *
                static_cast<double>(ts.tv_nsec)) - start;

#endif

    elapsed_time = duration*conv_factor;
  }

  inline void cpu_timer::tac(const std::string &msg) { tac(); print(msg); };

  void cpu_timer::print(const std::string &msg) {
    std::cout << msg << " "; print();
  }

  float cpu_timer::print() {
    if(elapsed_time) {
      std::cout << "elapsed time: " << elapsed_time << " seconds\n";
    }
	return elapsed_time;
  }

  void cpu_timer::reset() { start = 0; duration = 0; elapsed_time = 0; }
  double cpu_timer::getTime() { return elapsed_time; }


#if defined(_WINDOWS)
# undef WIN32_LEAN_AND_MEAN
#endif

#endif // TIMER_Hcpu_timer