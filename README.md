#ifndef ABENCHMARK_H
#define ABENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

#if defined(_MSC_VER)
#include <intrin.h>   // MSVC - For __rdtsc
#else
#include <x86intrin.h>  // GCC/Clang on x86
#endif

// Default CPU frequency if not provided
#ifndef CPU_FREQ
#define CPU_FREQ 3.0e9  // e.g., 3.0 GHz CPU
#endif

// Uncomment to use RDTSC instead of std::chrono
// #define CLOCK_SOURCE_RDTSC

#ifdef CLOCK_SOURCE_RDTSC
#define GET_TIME() __rdtsc()
#define CALCULATE_TIME_NS(end, start) \
  (static_cast<double>((end) - (start)) / CPU_FREQ * 1.0e9)
typedef uint64_t CLOCK_TYPE;
#else
#define GET_TIME() std::chrono::high_resolution_clock::now()
typedef std::chrono::time_point<std::chrono::high_resolution_clock> CLOCK_TYPE;
#define CALCULATE_TIME_NS(end, start) \
  (std::chrono::duration_cast<std::chrono::nanoseconds>((end) - (start)).count())
#endif

// Default number of runs if not specified by the user
#ifndef BENCH_NUMBER_OF_RUN
#define BENCH_NUMBER_OF_RUN 1000
#endif

namespace Abench {

/**
 * @class Mutex
 * @brief A simple RAII-friendly mutex wrapper.
 */
class Mutex {
 public:
  Mutex() = default;
  void lock()   { lock_guard_.lock(); }
  void unlock() { lock_guard_.unlock(); }

 private:
  std::mutex lock_guard_;
};

/**
 * @struct Stats
 * @brief Holds benchmark statistics for a single benchmark run.
 */
struct Stats {
  double min_ns;
  double max_ns;
  double mean_ns;
};

/**
 * @class State
 * @brief Stores iteration state and timing info for each iteration.
 */
class State {
 public:
  /**
   * @param n Number of iterations for this benchmark.
   */
  explicit State(int n)
      : iteration_count_(n), manual_(false) {
    stat_.min_ns = std::numeric_limits<double>::max();
    stat_.max_ns = 0.0;
    stat_.mean_ns = 0.0;
  }

  /**
   * @struct TimerProxy
   * @brief RAII object that measures time of a single iteration if manual mode is not enabled.
   */
  struct TimerProxy {
    TimerProxy(Stats& stat,
               bool manual,
               CLOCK_TYPE& start_manual_ref)
        : stat_(stat),
          manual_mode_(manual),
          start_manual_(start_manual_ref) {
      start_ = GET_TIME();
    }

    ~TimerProxy() {
      CLOCK_TYPE end_time = GET_TIME();
      if (!manual_mode_) {
        double ns = CALCULATE_TIME_NS(end_time, start_);
        if (ns < stat_.min_ns) stat_.min_ns = ns;
        if (ns > stat_.max_ns) stat_.max_ns = ns;
        stat_.mean_ns += ns;
      } else {
        // If manual mode, store start time for user to explicitly stop later
        start_manual_ = start_;
      }
    }

   private:
    Stats& stat_;
    bool manual_mode_;
    CLOCK_TYPE start_;
    CLOCK_TYPE& start_manual_;
  };

  /**
   * @class Iterator
   * @brief Custom iterator used in range-based for loops: for (auto _ : state) {...}
   */
  class Iterator {
   public:
    Iterator(int index, State& st)
        : index_(index), state_(st) {}

    bool operator!=(const Iterator& other) const {
      return index_ != other.index_;
    }

    Iterator& operator++() {
      ++index_;
      return *this;
    }

    TimerProxy operator*() {
      return TimerProxy(state_.stat_,
                        state_.manual_,
                        state_.start_manual_);
    }

   private:
    int index_;
    State& state_;
  };

  // Begin and end for range-based for
  Iterator begin() { return Iterator(0, *this); }
  Iterator end()   { return Iterator(iteration_count_, *this); }

  /**
   * @brief Begin manual timing for a single iteration block.
   */
  void StartTimer() {
    manual_ = true;
    start_manual_ = GET_TIME();
  }

  /**
   * @brief Stop manual timing for a single iteration block.
   */
  void StopTimer() {
    CLOCK_TYPE end_time = GET_TIME();
    double ns = CALCULATE_TIME_NS(end_time, start_manual_);
    if (ns < stat_.min_ns) stat_.min_ns = ns;
    if (ns > stat_.max_ns) stat_.max_ns = ns;
    stat_.mean_ns += ns;
  }

  /**
   * @brief Compute min, max, mean of the recorded durations.
   * @return Stats struct with min_ns, max_ns, mean_ns
   */
  Stats getStats() {
    stat_.mean_ns /= GetIterations();
    return stat_;
  }

  int GetIterations() const {
    return iteration_count_;
  }

  //--------------------------------------------------------------------------
  // Multi-dimensional Arg support
  //--------------------------------------------------------------------------
  void SetArgs(const std::vector<int>& args) {
    currentArgs_ = args;
  }

  int range(size_t i) const {
    if (i < currentArgs_.size()) {
      return currentArgs_[i];
    }
    return -1; // out of range
  }

 private:
  int iteration_count_;
  bool manual_;
  Stats stat_;
  CLOCK_TYPE start_manual_;

  // For multi-dimensional arguments
  std::vector<int> currentArgs_;
};

/**
 * @class Benchmark
 * @brief Base interface for all user-defined benchmarks.
 */
class Benchmark {
 public:
  explicit Benchmark(const std::string& name)
      : name_(name), iterations_(BENCH_NUMBER_OF_RUN),
        rangeMultiplier_(2) // default multiplier
  {}

  Benchmark* Name(const std::string& n) {
    name_ = n;
    return this;
  }

  Benchmark* Iterations(size_t it) {
    iterations_ = it;
    return this;
  }

  //--------------------------------------------------------------------------
  // Google Benchmark–style argument methods
  //--------------------------------------------------------------------------

  /**
   * @brief Single scalar argument => one-dimensional set
   */
  Benchmark* Arg(int a) {
    argSets_.push_back({a});
    return this;
  }

  /**
   * @brief Multi-argument set => e.g. Args({32, 64}) means two-dim arguments
   */
  Benchmark* Args(const std::initializer_list<int>& list) {
    argSets_.push_back(std::vector<int>(list));
    return this;
  }

  /**
   * @brief Range-based arguments. Repeatedly multiply `start` by `rangeMultiplier_`
   * until we exceed `limit`.
   *
   * Example: If start=8, limit=1024, rangeMultiplier_=2,
   * we get 8,16,32,64,128,256,512,1024
   */
  Benchmark* Range(int start, int limit) {
    int val = start;
    while (val <= limit) {
      Arg(val); // pushes {val}
      val *= rangeMultiplier_;
    }
    return this;
  }

  /**
   * @brief For enumerating a dense sequence [start..limit] in increments of step.
   * Example: DenseRange(1,4) => 1,2,3,4
   */
  Benchmark* DenseRange(int start, int limit, int step = 1) {
    for (int i = start; i <= limit; i += step) {
      Arg(i);
    }
    return this;
  }

  /**
   * @brief Set the multiplier used by Range().
   */
  Benchmark* RangeMultiplier(int mul) {
    rangeMultiplier_ = (mul > 1) ? mul : 2; // avoid nonsense
    return this;
  }

  //--------------------------------------------------------------------------
  // Accessors
  //--------------------------------------------------------------------------

  const std::string& GetName() const {
    return name_;
  }

  size_t GetIterations() const {
    return iterations_;
  }

  // Return all argument sets (possibly empty if user never sets them)
  const std::vector<std::vector<int>>& GetAllArgSets() const {
    return argSets_;
  }

  bool HasArgSets() const {
    return !argSets_.empty();
  }

  virtual void Run(State& st) = 0;
  virtual ~Benchmark() {}

 private:
  std::string name_;
  size_t iterations_;

  // Arg data
  std::vector<std::vector<int>> argSets_;
  int rangeMultiplier_;
};

// Function pointer signature
typedef void (Function)(State&);

/**
 * @class FunctionBenchmark
 * @brief Invokes a user-provided function for each iteration.
 */
class FunctionBenchmark : public Benchmark {
 public:
  FunctionBenchmark(const std::string& name, Function* func)
      : Benchmark(name), func_(func) {}

  void Run(State& st) override {
    func_(st);
  }

 private:
  Function* func_;
};

/**
 * @class BenchmarkRegister
 * @brief Singleton that stores all benchmarks and provides a place to run them.
 */
class BenchmarkRegister {
 public:
  static BenchmarkRegister* GetInstance() {
    static BenchmarkRegister instance;
    return &instance;
  }

  Benchmark* AddBenchmark(std::unique_ptr<Benchmark> b) {
    mutex_.lock();
    benchmarks_.push_back(std::move(b));
    mutex_.unlock();
    return benchmarks_.back().get();
  }

  void Clear() {
    mutex_.lock();
    benchmarks_.clear();
    benchmarks_.shrink_to_fit();
    mutex_.unlock();
  }

  std::vector<std::unique_ptr<Benchmark>> benchmarks_;

 private:
  BenchmarkRegister() {}
  Mutex mutex_;
};

/**
 * @brief Helper to register a new benchmark (ownership transferred).
 */
inline Benchmark* RegisterBenchmarkInternal(Benchmark* b) {
  std::unique_ptr<Benchmark> ptr(b);
  return BenchmarkRegister::GetInstance()->AddBenchmark(std::move(ptr));
}

/**
 * @brief Initialize the benchmark environment (e.g., set thread affinity).
 */
inline void Initialize() {
  // Example: pin thread to CPU #2
  HANDLE thread = GetCurrentThread();
  DWORD_PTR affinity_mask = 4; // CPU #2 on Windows
  DWORD_PTR result = SetThreadAffinityMask(thread, affinity_mask);
  if (result == 0) {
    std::cerr << "Failed to set thread affinity. Error: "
              << GetLastError() << std::endl;
  } else {
    std::cout << "Thread affinity set to CPU #2 (mask=4).\n";
  }
}

/**
 * @struct BenchResult
 * @brief Holds final stats for printing or exporting.
 */
struct BenchResult {
  std::string name;
  size_t iterations;
  double min_ns;
  double max_ns;
  double mean_ns;
};

/**
 * @enum OutputFormat
 * @brief Specifies the type of file output to produce, if any.
 */
enum class OutputFormat {
  None,
  JSON,
  CSV
};

// Global output config
static OutputFormat g_format = OutputFormat::None;
static std::string g_fileName;

/**
 * @brief User sets the output format ("csv" or "json") and file name.
 *        If format is neither "csv" nor "json", we disable file output.
 */
inline void SetOutputFormat(const std::string& format, const std::string& fileName) {
  if (format == "csv") {
    g_format   = OutputFormat::CSV;
    g_fileName = fileName;
  } else if (format == "json") {
    g_format   = OutputFormat::JSON;
    g_fileName = fileName;
  } else {
    // No recognized format => no file output
    g_format   = OutputFormat::None;
    g_fileName.clear();
  }
}

/**
 * @brief Writes results in JSON format to the given stream.
 */
inline void WriteJson(std::ostream& os, const std::vector<BenchResult>& results) {
  os << "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const BenchResult& r = results[i];
    os << "  {\n"
       << "    \"Benchmark\": \"" << r.name << "\",\n"
       << "    \"Iterations\": " << r.iterations << ",\n"
       << "    \"Min(ns)\": " << std::fixed << std::setprecision(2) << r.min_ns << ",\n"
       << "    \"Max(ns)\": " << r.max_ns << ",\n"
       << "    \"Mean(ns)\": " << r.mean_ns << "\n"
       << "  }";
    if (i + 1 < results.size()) {
      os << ",";
    }
    os << "\n";
  }
  os << "]\n";
}

/**
 * @brief Writes results in CSV format to the given stream.
 */
inline void WriteCsv(std::ostream& os, const std::vector<BenchResult>& results) {
  os << "Benchmark,Iterations,Min(ns),Max(ns),Mean(ns)\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const BenchResult& r = results[i];
    if (r.mean_ns <= 0.0) {
      os << r.name << "," << r.iterations << ",-,-,-\n";
    } else {
      os << r.name << "," << r.iterations << ","
         << std::fixed << std::setprecision(2) << r.min_ns << ","
         << r.max_ns << "," << r.mean_ns << "\n";
    }
  }
}

/**
 * @brief Print the results in a table (Google Benchmark–like),
 *        using dynamic column widths for large numeric values.
 */
inline void PrintTable(const std::vector<BenchResult>& results) {
  // Column headers
  const std::string titleName = "Benchmark";
  const std::string titleIter = "Iterations";
  const std::string titleMin  = "Min(ns)";
  const std::string titleMax  = "Max(ns)";
  const std::string titleMean = "Mean(ns)";

  // We'll convert each row to strings, track max widths.
  struct Row {
    std::string name;
    std::string iter;
    std::string minv;
    std::string maxv;
    std::string meanv;
  };

  // Store rows
  std::vector<Row> table;
  table.reserve(results.size());

  // Track max column width for each column
  size_t maxNameWidth  = titleName.size();
  size_t maxIterWidth  = titleIter.size();
  size_t maxMinWidth   = titleMin.size();
  size_t maxMaxWidth   = titleMax.size();
  size_t maxMeanWidth  = titleMean.size();

  // Helper to format a double with 2 decimals
  auto fmt = [](double val) -> std::string {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << val;
    return oss.str();
  };

  // Build table rows
  for (size_t i = 0; i < results.size(); ++i) {
    const BenchResult& r = results[i];

    Row row;
    // 1) Benchmark name
    row.name = r.name;
    if (row.name.size() > maxNameWidth) {
      maxNameWidth = row.name.size();
    }

    // 2) Iterations
    row.iter = std::to_string(r.iterations);
    if (row.iter.size() > maxIterWidth) {
      maxIterWidth = row.iter.size();
    }

    // 3) Min, Max, Mean
    if (r.mean_ns <= 0.0) {
      row.minv  = "-";
      row.maxv  = "-";
      row.meanv = "-";
    } else {
      row.minv  = fmt(r.min_ns);
      row.maxv  = fmt(r.max_ns);
      row.meanv = fmt(r.mean_ns);

      if (row.minv.size()  > maxMinWidth)  maxMinWidth  = row.minv.size();
      if (row.maxv.size()  > maxMaxWidth)  maxMaxWidth  = row.maxv.size();
      if (row.meanv.size() > maxMeanWidth) maxMeanWidth = row.meanv.size();
    }
    table.push_back(row);
  }

  // A small padding between columns
  const size_t pad = 2;
  // Compute total line width
  size_t totalWidth =
      maxNameWidth + maxIterWidth + maxMinWidth + maxMaxWidth + maxMeanWidth
      + (4 * pad);
  // Minimum for aesthetics
  if (totalWidth < 80) {
    totalWidth = 80;
  }

  // Helper to print a separator line
  auto printSep = [&](std::ostream& os) {
    os << std::string(totalWidth, '-') << "\n";
  };

  // Print table
  printSep(std::cout);

  // Print header
  // Benchmark name is left-aligned, numeric columns are right-aligned
  std::cout
    << std::left  << std::setw(static_cast<int>(maxNameWidth)) << titleName
    << std::string(pad, ' ')
    << std::right << std::setw(static_cast<int>(maxIterWidth)) << titleIter
    << std::string(pad, ' ')
    << std::right << std::setw(static_cast<int>(maxMinWidth))  << titleMin
    << std::string(pad, ' ')
    << std::right << std::setw(static_cast<int>(maxMaxWidth))  << titleMax
    << std::string(pad, ' ')
    << std::right << std::setw(static_cast<int>(maxMeanWidth)) << titleMean
    << "\n";

  printSep(std::cout);

  // Print rows
  for (size_t i = 0; i < table.size(); ++i) {
    const Row& row = table[i];
    std::cout
      << std::left  << std::setw(static_cast<int>(maxNameWidth)) << row.name
      << std::string(pad, ' ')
      << std::right << std::setw(static_cast<int>(maxIterWidth)) << row.iter
      << std::string(pad, ' ')
      << std::right << std::setw(static_cast<int>(maxMinWidth))  << row.minv
      << std::string(pad, ' ')
      << std::right << std::setw(static_cast<int>(maxMaxWidth))  << row.maxv
      << std::string(pad, ' ')
      << std::right << std::setw(static_cast<int>(maxMeanWidth)) << row.meanv
      << "\n";
  }

  printSep(std::cout);
}

/**
 * @brief Run all registered benchmarks, print table to stdout, and optionally
 *        write JSON or CSV results to a file if the user specified an output format.
 */
inline void RunAll() {
  BenchmarkRegister* instance = BenchmarkRegister::GetInstance();
  std::vector<BenchResult> results;

  // In the worst case, each benchmark can have many arg sets
  // Reserve something, though it's not a big deal if it re-allocates
  results.reserve(instance->benchmarks_.size() * 4);

  // 1) Run each benchmark, gather stats
  for (auto& benchPtr : instance->benchmarks_) {
    Benchmark* bench = benchPtr.get();

    // If the benchmark has argument sets:
    if (bench->HasArgSets()) {
      const auto& allArgSets = bench->GetAllArgSets();
      for (const auto& argSet : allArgSets) {
        // Create State
        State st(static_cast<int>(bench->GetIterations()));
        st.SetArgs(argSet);

        // Run the user function
        bench->Run(st);
        Stats s = st.getStats();

        // Build a result record, e.g. "Name/2" or "Name/2/4"
        std::ostringstream oss;
        oss << bench->GetName();
        for (int v : argSet) {
          oss << "/" << v;
        }

        BenchResult r;
        r.name       = oss.str();
        r.iterations = bench->GetIterations();
        r.min_ns     = s.min_ns;
        r.max_ns     = s.max_ns;
        r.mean_ns    = s.mean_ns;

        results.push_back(r);
      }
    } else {
      // No arguments => old single-run approach
      State st(static_cast<int>(bench->GetIterations()));
      bench->Run(st);
      Stats s = st.getStats();

      BenchResult r;
      r.name       = bench->GetName();
      r.iterations = bench->GetIterations();
      r.min_ns     = s.min_ns;
      r.max_ns     = s.max_ns;
      r.mean_ns    = s.mean_ns;

      results.push_back(r);
    }
  }

  // 2) Print dynamic table to stdout
  PrintTable(results);

  // 3) If user specified a format + file name, write to that file
  if (g_format != OutputFormat::None && !g_fileName.empty()) {
    std::ofstream ofs(g_fileName.c_str());
    if (!ofs) {
      std::cerr << "Error: Unable to open file '" << g_fileName
                << "' for writing.\n";
      return;
    }
    switch (g_format) {
      case OutputFormat::CSV:
        WriteCsv(ofs, results);
        break;
      case OutputFormat::JSON:
        WriteJson(ofs, results);
        break;
      default:
        break;
    }
    ofs.close();
  }
}

// -----------------------------------------------------------------------------
// Macros for registering benchmarks
// -----------------------------------------------------------------------------
#ifdef __GNUC__
#define BENCHMARK_UNUSED __attribute__((unused))
#else
#define BENCHMARK_UNUSED
#endif

// We use __LINE__ to generate unique variable names in C++14
#define BENCHMARK_PRIVATE_CONCAT(a, b) a##b
#define BENCHMARK_PRIVATE_NAME(line) BENCHMARK_PRIVATE_CONCAT(benchmark_unique_, line)

/**
 * @brief Create and register a FunctionBenchmark.
 * Usage:
 *   BENCHMARK(MyFunction)
 *      ->Name("Test")
 *      ->Iterations(5)
 *      ->Arg(1)
 *      ->Arg(2)
 *      ->Range(1,1024)
 *      ->DenseRange(0,10)
 *      ->Args({32, 64})
 *      ->RangeMultiplier(4);
 */
#define BENCHMARK(func) \
  static ::Abench::Benchmark* BENCHMARK_PRIVATE_NAME(__LINE__) BENCHMARK_UNUSED = \
      ::Abench::RegisterBenchmarkInternal( \
          new ::Abench::FunctionBenchmark(#func, func))

}  // namespace Abench

#endif  // ABENCHMARK_H

























#include "ABenchmark.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

//------------------------------------------------------------------------------
// Example 1: Automatic Timing - Summation of a vector
//------------------------------------------------------------------------------
void BM_SumVector(Abench::State& st) {
  // Use the first argument (dimension 0) as the vector size
  int size = st.range(0);

  // Possibly a second argument if specified
  int secondVal = st.range(1); // -1 if not set

  // Prepare a vector of 'size' elements
  std::vector<int> data(size, 1);

  // Range-based for automatically times each iteration
  for (auto _ : st) {
    long long sum = 0;
    for (int x : data) {
      sum += x;
    }

    // Just to simulate using the second argument, we might do:
    // sum += secondVal;  // If secondVal != -1
  }
}

//------------------------------------------------------------------------------
// Example 2: Manual Timing - Sleep for demonstration
//------------------------------------------------------------------------------
void BM_ManualSleep(Abench::State& st) {
  for (int i = 0; i < st.GetIterations(); ++i) {
    st.StartTimer();
    // Simulate work with a 1 ms sleep
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    st.StopTimer();
  }
}

//------------------------------------------------------------------------------
// Example 3: Automatic Timing but trivial work
//------------------------------------------------------------------------------
void BM_Trivial(Abench::State& st) {
  for (auto _ : st) {
    // No real work, just a placeholder
    int x = 42;
    (void)x; // avoid unused variable warning
  }
}

//------------------------------------------------------------------------------
// Register Benchmarks
//------------------------------------------------------------------------------
/**
 * 1) Demonstrates Arg, Range, DenseRange, RangeMultiplier, and multi-argument sets
 */
BENCHMARK(BM_SumVector)
    ->Name("SumVector")
    ->Iterations(50)
    // Single Arg => st.range(0) = 1000
    ->Arg(1000)
    // Range from 8 to 64, multiplying by default=2: 8,16,32,64
    ->Range(8, 64)
    // DenseRange from 1..4 => 1,2,3,4
    ->DenseRange(1, 4,5)
    // Example of RangeMultiplier usage:
    ->RangeMultiplier(4)
    // Another Range from 16..256 but in powers of 4 => 16,64,256
    ->Range(16, 256)
    // multi-dimensional argument => st.range(0)=9999, st.range(1)=5555
    ->Args({9999, 5555,5, 4});

/**
 * 2) Demonstrates manual StartTimer/StopTimer
 */
BENCHMARK(BM_ManualSleep)
    ->Name("ManualSleep")
    ->Iterations(10);

/**
 * 3) Very trivial auto-timed function
 */
BENCHMARK(BM_Trivial)
    ->Name("Trivial")
    ->Iterations(100);

//------------------------------------------------------------------------------
// main()
//------------------------------------------------------------------------------
int main() {
  // Optional: set output format, e.g. CSV => "results.csv"
  Abench::SetOutputFormat("csv", "results.csv");

  // Pin this thread to CPU #2 (as in your Abench's Initialize())
  Abench::Initialize();

  // Run all benchmarks
  Abench::RunAll();

  return 0;
}

