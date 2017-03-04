#include "caffe/profiler.hpp"

#if defined(_MSC_VER) && _MSC_VER <= 1800
#include <Windows.h>
#else
#include <chrono>
#endif  // _MSC_VER

#include <fstream>

namespace caffe {

Profiler::Profiler()
    : init_(Now()), state_(kNotRunning) {
  scope_stack_.reserve(10);
  scopes_.reserve(1024);
}

Profiler *Profiler::Get() {
  static Profiler inst;
  return &inst;
}

void Profiler::ScopeStart(const char *name) {
  if (state_ == kNotRunning) return;
  ScopePtr scope(new Scope);
  if (!scope_stack_.empty()) {
    scope->name = scope_stack_.back()->name + ":" + name;
  }
  else{
    scope->name = name;
  }
  scope->start_microsec = Now() - init_;
  scope_stack_.push_back(scope);
}

void Profiler::ScopeEnd() {
  if (state_ == kNotRunning) return;
  CHECK(!scope_stack_.empty());
  ScopePtr current_scope = scope_stack_.back();
  current_scope->end_microsec = Now() - init_;
  scopes_.push_back(current_scope);
  // pop stack
  scope_stack_.pop_back();
}

uint64_t Profiler::Now() const {
#if defined(_MSC_VER) && _MSC_VER <= 1800
  LARGE_INTEGER frequency, counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return counter.QuadPart * 1000000 / frequency.QuadPart;
#else
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif  // _MSC_VER
}

static void ProfilerWriteEvent(std::ofstream &file,
                              const char *name,
                              const char *ph,
                              uint64_t ts) {
  file << "    {" << std::endl;
  file << "      \"name\": \"" << name << "\"," << std::endl;
  file << "      \"cat\": \"category\"," << std::endl;
  file << "      \"ph\": \"" << ph << "\"," << std::endl;
  file << "      \"ts\": " << ts << "," << std::endl;
  file << "      \"pid\": 0," << std::endl;
  file << "      \"tid\": 0" << std::endl;
  file << "    }";
}

void Profiler::DumpProfile(const char *fn) const {
  CHECK(scope_stack_.empty());
  CHECK_EQ(state_, kNotRunning);

  std::ofstream file;
  file.open(fn);
  file << "{" << std::endl;
  file << "  \"traceEvents\": [";

  bool is_first = true;
  for (auto scope : scopes_) {
    if (is_first) {
      file << std::endl;
      is_first = false;
    }
    else {
      file << "," << std::endl;
    }
    ProfilerWriteEvent(file, scope->name.c_str(), "B", scope->start_microsec);
    file << "," << std::endl;
    ProfilerWriteEvent(file, scope->name.c_str(), "E", scope->end_microsec);
  }

  file << "  ]," << std::endl;
  file << "  \"displayTimeUnit\": \"ms\"" << std::endl;
  file << "}" << std::endl;
}

}  // namespace caffe
