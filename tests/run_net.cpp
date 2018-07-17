#include <random>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>

#include <caffe/net.hpp>
#include <caffe/profiler.hpp>

using namespace std;
using namespace caffe;

struct Mat {
  int rows_, cols_;
  real_t *data_;

  Mat()
    : rows_(0), cols_(0), data_(nullptr) {
  }
  Mat(int rows, int cols)
    : rows_(rows), cols_(cols), data_(new real_t[rows*cols]) {
  }
  Mat(const Mat &m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    int size = rows_ * cols_;
    data_ = new real_t[size];
    memcpy(data_, m.data_, size * sizeof(real_t));
  }
  Mat(Mat &&m)
    : data_(m.data_) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    m.data_ = nullptr;
  }
  Mat &operator=(const Mat &m) {
    if (this == &m) return *this;
    rows_ = m.rows_;
    cols_ = m.cols_;
    int size = rows_ * cols_;
    if (data_) delete[] data_;
    data_ = new real_t[size];
    memcpy(data_, m.data_, size * sizeof(real_t));
    return *this;
  }
  Mat &operator=(Mat &&m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    std::swap(data_, m.data_);
    return *this;
  }
  ~Mat() {
    if (data_) delete[] data_;
  }

  real_t At(int r, int c) { return data_[r*cols() + c]; }
  int rows() { return rows_; }
  int cols() { return cols_; }
  float *data() { return data_; }

  static Mat Random(int rows, int cols, float scale=1) {
    Mat m(rows, cols);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<real_t> urd(-scale, scale);
    std::generate(m.data_, m.data_ + m.rows_*m.cols_, [&]() { return urd(dre); });
    return m;
  }
};

/*! \brief Timer */
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /*! \brief start or restart timer */
  inline void Tic() {
    start_ = Clock::now();
  }
  /*! \brief stop timer */
  inline void Toc() {
    end_ = Clock::now();
  }
  /*! \brief return time in ms */
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }
private:
  Clock::time_point start_, end_;
};

struct TestFunctor {
  std::shared_ptr<caffe::Net> net;

  TestFunctor(const std::string &prototxt, const std::string &caffemodel)
    : net(new caffe::Net(prototxt)) {
    shared_ptr<caffe::NetParameter> np = caffe::ReadBinaryNetParameterFromFile(caffemodel);
    net->CopyTrainedLayersFrom(*np);
  }
  /*!
   * \brief generate input data
   * \return feature maps
   */
  virtual std::vector<Mat> Generate() = 0;
  /*!
   * \brief forward network
   * \param img input image
   */
  void Forward() {
    // transform input image to feature map
    auto data = this->Generate();
    std::shared_ptr<Blob> input = this->net->blob_by_name("data");
    CHECK_GT(data.size(), 1);
    CHECK_EQ(1, input->num());
    CHECK_EQ(data.size(), input->channels());
    CHECK_EQ(data[0].rows(), input->height());
    CHECK_EQ(data[0].cols(), input->width());
    // copy to network input buffer
    const int bias = input->offset(0, 1);
    const int bytes = bias * sizeof(real_t);
    for (int k = 0; k < data.size(); k++) {
      memcpy(input->mutable_cpu_data() + k * bias, data[k].data(), bytes);
    }
    // forward network
    this->net->Forward();
  }
};

struct NIN : public TestFunctor {
  NIN(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

struct GoogLeNet : public TestFunctor {
  GoogLeNet(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

struct ResNet : public TestFunctor {
  ResNet(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

void thread_test();
void test_io();
void test_reshape();

int main(int argc, char *argv[]) {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }

  Timer timer;
  Profiler *profiler = Profiler::Get();
  profiler->TurnON();

  // test nin, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#network-in-network-model
  {
    LOG(INFO) << "Test NIN";
    auto test = NIN("model/nin.prototxt", "model/nin.caffemodel");
    timer.Tic();
    profiler->ScopeStart("nin");
    test.Forward();
    profiler->ScopeEnd();
    timer.Toc();
    LOG(INFO) << "Forward NIN costs " << timer.Elasped() << " ms";
  }
  // test googlenet, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing
  {
    LOG(INFO) << "Test GoogLeNet";
    auto test = GoogLeNet("model/googlenet.prototxt", "model/googlenet.caffemodel");
    timer.Tic();
    profiler->ScopeStart("googlenet");
    test.Forward();
    profiler->ScopeEnd();
    timer.Toc();
    LOG(INFO) << "Forward GoogLeNet costs " << timer.Elasped() << " ms";
  }
  // test resnet, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#imagenet-pre-trained-models-with-batch-normalization
  {
    LOG(INFO) << "Test ResNet";
    auto test = ResNet("model/resnet.prototxt", "model/resnet.caffemodel");
    timer.Tic();
    profiler->ScopeStart("resnet");
    test.Forward();
    profiler->ScopeEnd();
    timer.Toc();
    LOG(INFO) << "Forward ResNet costs " << timer.Elasped() << " ms";
  }

  // dump profile data
  profiler->TurnOFF();
  profiler->DumpProfile("profile.json");

  // test multi-thread
  const int kThreads = 3;
  LOG(INFO) << "test " << kThreads << " threads";
  std::vector<std::thread> pool;
  for (int i = 0; i < kThreads; i++) {
    pool.emplace_back(std::thread{thread_test});
  }
  for (int i = 0; i < kThreads; i++) {
    pool[i].join();
  }

  test_io();
  test_reshape();
  return 0;
}

void thread_test() {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }
  LOG(INFO) << "Test ResNet";
  auto test = ResNet("model/resnet.prototxt", "model/resnet.caffemodel");
  test.Forward();
  caffe::MemPoolClear();
}

void test_io() {
  // test IO
  ifstream fin("model/resnet.prototxt");
  stringstream buffer;
  buffer << fin.rdbuf();
  fin.close();
  string prototxt = buffer.str();
  shared_ptr<NetParameter> network_param = ReadTextNetParameterFromBuffer(prototxt.c_str(), prototxt.length());
  fin.open("model/resnet.caffemodel", ios::binary);
  buffer.str("");
  buffer.clear();
  buffer << fin.rdbuf();
  fin.close();
  string caffemodel = buffer.str();
  shared_ptr<NetParameter> model_param = ReadBinaryNetParameterFromBuffer(caffemodel.c_str(), caffemodel.length());
  Net net(*network_param);
  net.CopyTrainedLayersFrom(*model_param);

  MemPoolState st = caffe::MemPoolGetState();
  auto __Calc__ = [](int size) -> double {
    return round(static_cast<double>(size) / (1024 * 1024) * 100) / 100;
  };
  LOG(INFO) << "[CPU] Hold " << __Calc__(st.cpu_mem) << " M, Not Uses " << __Calc__(st.unused_cpu_mem) << " M";
  LOG(INFO) << "[GPU] Hold " << __Calc__(st.gpu_mem) << " M, Not Uses " << __Calc__(st.unused_gpu_mem) << " M";
}

void test_reshape() {
  LOG(INFO) << "Test Reshape";
  auto resnet = ResNet("model/resnet.prototxt", "model/resnet.caffemodel");
  std::vector<Mat> fm;
  fm.push_back(Mat::Random(224, 224, 128));
  fm.push_back(Mat::Random(224, 224, 128));
  fm.push_back(Mat::Random(224, 224, 128));
  std::shared_ptr<Blob> input = resnet.net->blob_by_name("data");
  std::shared_ptr<Blob> prob = resnet.net->blob_by_name("prob");

  auto forward_bs = [&](int bs) {
    input->Reshape(bs, 3, 224, 224);
    const int bias = input->offset(0, 1);
    const int bytes = bias * sizeof(real_t);
    for (int k = 0; k < bs * fm.size(); k++) {
      memcpy(input->mutable_cpu_data() + k * bias, fm[k % fm.size()].data(), bytes);
    }
    resnet.net->Forward();
  };

  Mat b1_prob(1, 1000);
  Mat b2_prob(2, 1000);

  auto check_result = [&]() {
    auto AllMostEQ = [](float x, float y) {
      return std::abs(x - y) < 1e-6;
    };
    for (int i = 0; i < 1000; i++) {
      float p1 = b1_prob.At(0, i);
      float p2 = b2_prob.At(0, i);
      float p3 = b2_prob.At(1, i);
      CHECK(AllMostEQ(p1, p2) && AllMostEQ(p2, p3)) << p1 << " " << p2 << " " << p3;
    }
  };

  // batch size 1
  forward_bs(1);
  memcpy(b1_prob.data_, prob->cpu_data(), 1000 * sizeof(real_t));
  // batch size 2
  forward_bs(2);
  memcpy(b2_prob.data_, prob->cpu_data(), 2000 * sizeof(real_t));
  check_result();
  // batch size 1
  forward_bs(1);
  memcpy(b1_prob.data_, prob->cpu_data(), 1000 * sizeof(real_t));
  check_result();
  caffe::MemPoolClear();
}
