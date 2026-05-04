#include <grpcpp/grpcpp.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <malloc.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "ortserver.pb.h"
#include "ortserver.grpc.pb.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct InputStorage {
  std::vector<float> fp32;
  std::vector<Ort::Float16_t> fp16;
  std::vector<int32_t> int32;
  std::vector<int64_t> int64;
  std::vector<uint8_t> uint8;
};

struct LoadedModel {
  std::string name;
  int64_t version = 0;
  fs::path model_path;
  mutable Ort::Session session{nullptr};
  std::vector<std::string> input_names;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<std::string> output_names;
  std::vector<ONNXTensorElementDataType> output_types;
};

struct ModelKey {
  std::string name;
  int64_t version = 0;

  bool operator<(const ModelKey& other) const {
    if (name != other.name) {
      return name < other.name;
    }
    return version < other.version;
  }
};

struct LogField {
  std::string key;
  std::string value;
};

static std::string now_utc_iso8601() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << ms.count() << 'Z';
  return out.str();
}

static std::string quote_value(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char ch : value) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

static void log_line(const std::string& level, const std::string& event,
                     std::initializer_list<LogField> fields = {}) {
  std::ostringstream out;
  out << "ts=" << now_utc_iso8601() << " level=" << level << " event=" << event;
  for (const auto& field : fields) {
    out << ' ' << field.key << '=' << quote_value(field.value);
  }
  std::cout << out.str() << std::endl;
}

static void log_error(const std::string& event, std::initializer_list<LogField> fields = {}) {
  std::ostringstream out;
  out << "ts=" << now_utc_iso8601() << " level=error event=" << event;
  for (const auto& field : fields) {
    out << ' ' << field.key << '=' << quote_value(field.value);
  }
  std::cerr << out.str() << std::endl;
}

static std::string escape_prom_label(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char ch : value) {
    switch (ch) {
      case '\\':
      case '"':
        out.push_back('\\');
        out.push_back(ch);
        break;
      case '\n':
        out.append("\\n");
        break;
      case '\r':
        out.append("\\r");
        break;
      case '\t':
        out.append("\\t");
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  out.push_back('"');
  return out;
}

static int64_t current_rss_bytes() {
  std::ifstream statm("/proc/self/statm");
  long pages = 0;
  long resident = 0;
  if (!(statm >> pages >> resident)) {
    return 0;
  }
  long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return 0;
  }
  return static_cast<int64_t>(resident) * static_cast<int64_t>(page_size);
}

static int64_t now_micros(const Clock::time_point& started) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count();
}

template <typename T, typename Repeated>
static std::vector<T> repeated_to_vector(const Repeated& repeated) {
  return std::vector<T>(repeated.begin(), repeated.end());
}

template <typename T>
static std::vector<T> bytes_to_vector(const std::string& bytes) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error("tensor_content size is not aligned to element size");
  }
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

static std::vector<int64_t> shape_to_dims(const ortserver::v1::Tensor& tensor, size_t fallback) {
  std::vector<int64_t> dims = repeated_to_vector<int64_t>(tensor.shape());
  if (dims.empty()) {
    dims = {static_cast<int64_t>(fallback)};
  }
  size_t total = 1;
  for (auto dim : dims) {
    if (dim < 0) {
      throw std::runtime_error("invalid negative dimension");
    }
    total *= static_cast<size_t>(dim);
  }
  if (total != fallback) {
    throw std::runtime_error("shape does not match tensor element count");
  }
  return dims;
}

static std::vector<fs::path> discover_model_paths(const fs::path& model_root) {
  if (!fs::exists(model_root)) {
    throw std::runtime_error("model root does not exist: " + model_root.string());
  }

  std::vector<fs::path> paths;
  for (const auto& model_dir : fs::directory_iterator(model_root)) {
    if (!model_dir.is_directory()) {
      continue;
    }
    for (const auto& version_dir : fs::directory_iterator(model_dir)) {
      if (!version_dir.is_directory()) {
        continue;
      }
      auto model_path = version_dir.path() / "model.onnx";
      if (fs::exists(model_path)) {
        paths.push_back(model_path);
      }
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

static std::pair<std::string, int64_t> parse_model_name_version(const fs::path& model_path) {
  auto version_dir = model_path.parent_path();
  auto model_dir = version_dir.parent_path();
  auto model_name = model_dir.filename().string();
  auto version_str = version_dir.filename().string();
  int64_t version = std::stoll(version_str);
  return {model_name, version};
}

static std::set<ModelKey> discover_model_keys(const fs::path& model_root) {
  std::set<ModelKey> keys;
  for (const auto& model_path : discover_model_paths(model_root)) {
    auto [name, version] = parse_model_name_version(model_path);
    keys.insert(ModelKey{name, version});
  }
  return keys;
}

static int default_intra_threads() {
  auto cores = std::thread::hardware_concurrency();
  if (cores == 0) {
    return 1;
  }
  if (cores <= 2) {
    return static_cast<int>(cores);
  }
  return static_cast<int>(cores / 2);
}

static std::vector<int64_t> resolve_shape(const std::vector<int64_t>& shape, size_t count) {
  if (!shape.empty()) {
    size_t total = 1;
    for (auto dim : shape) {
      if (dim < 0) {
        throw std::runtime_error("invalid negative dimension");
      }
      total *= static_cast<size_t>(dim);
    }
    if (total != count) {
      throw std::runtime_error("shape does not match tensor value count");
    }
    return shape;
  }
  return {static_cast<int64_t>(count)};
}

struct RequestMetricKey {
  std::string model;
  std::string version;
  std::string status;

  bool operator<(const RequestMetricKey& other) const {
    if (model != other.model) {
      return model < other.model;
    }
    if (version != other.version) {
      return version < other.version;
    }
    return status < other.status;
  }
};

struct ErrorMetricKey {
  std::string model;
  std::string version;
  std::string error;

  bool operator<(const ErrorMetricKey& other) const {
    if (model != other.model) {
      return model < other.model;
    }
    if (version != other.version) {
      return version < other.version;
    }
    return error < other.error;
  }
};

class MetricsRegistry {
 public:
  void ObserveRequest(const std::string& model, const std::string& version, const std::string& status,
                      int64_t duration_us, const std::string& error = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    RequestMetricKey key{model, version, status};
    requests_total_[key] += 1;
    duration_sum_us_[key] += duration_us;
    duration_count_[key] += 1;
    auto& buckets = duration_buckets_[key];
    if (buckets.empty()) {
      buckets.resize(duration_bounds_seconds_.size() + 1, 0);
    }
    const double duration_seconds = static_cast<double>(duration_us) / 1000000.0;
    for (size_t i = 0; i < duration_bounds_seconds_.size(); ++i) {
      if (duration_seconds <= duration_bounds_seconds_[i]) {
        buckets[i] += 1;
      }
    }
    buckets.back() += 1;
    if (!error.empty()) {
      ErrorMetricKey error_key{model, version, error};
      errors_total_[error_key] += 1;
    }
  }

  std::string Render() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "# HELP ort_model_requests_total Total model requests.\n";
    out << "# TYPE ort_model_requests_total counter\n";
    for (const auto& [key, count] : requests_total_) {
      out << "ort_model_requests_total{model=" << escape_prom_label(key.model)
          << ",version=" << escape_prom_label(key.version)
          << ",status=" << escape_prom_label(key.status) << "} " << count << "\n";
    }

    out << "# HELP ort_model_request_duration_seconds Request duration in seconds.\n";
    out << "# TYPE ort_model_request_duration_seconds histogram\n";
    for (const auto& [key, buckets] : duration_buckets_) {
      auto sum_it = duration_sum_us_.find(key);
      auto count_it = duration_count_.find(key);
      double sum_seconds = sum_it == duration_sum_us_.end() ? 0.0
                                                            : static_cast<double>(sum_it->second) / 1000000.0;
      int64_t count = count_it == duration_count_.end() ? 0 : count_it->second;
      for (size_t i = 0; i < duration_bounds_seconds_.size(); ++i) {
        out << "ort_model_request_duration_seconds_bucket{model=" << escape_prom_label(key.model)
            << ",version=" << escape_prom_label(key.version)
            << ",status=" << escape_prom_label(key.status)
            << ",le=\"" << duration_bounds_seconds_[i] << "\"} " << buckets[i] << "\n";
      }
      out << "ort_model_request_duration_seconds_bucket{model=" << escape_prom_label(key.model)
          << ",version=" << escape_prom_label(key.version)
          << ",status=" << escape_prom_label(key.status) << ",le=\"+Inf\"} " << buckets.back() << "\n";
      out << "ort_model_request_duration_seconds_sum{model=" << escape_prom_label(key.model)
          << ",version=" << escape_prom_label(key.version)
          << ",status=" << escape_prom_label(key.status) << "} " << std::fixed << std::setprecision(6) << sum_seconds
          << "\n";
      out << "ort_model_request_duration_seconds_count{model=" << escape_prom_label(key.model)
          << ",version=" << escape_prom_label(key.version)
          << ",status=" << escape_prom_label(key.status) << "} " << count << "\n";
    }

    out << "# HELP ort_model_request_errors_total Total failed model requests.\n";
    out << "# TYPE ort_model_request_errors_total counter\n";
    for (const auto& [key, count] : errors_total_) {
      out << "ort_model_request_errors_total{model=" << escape_prom_label(key.model)
          << ",version=" << escape_prom_label(key.version)
          << ",error=" << escape_prom_label(key.error) << "} " << count << "\n";
    }
    return out.str();
  }

 private:
  mutable std::mutex mutex_;
  std::map<RequestMetricKey, int64_t> requests_total_;
  std::map<RequestMetricKey, int64_t> duration_sum_us_;
  std::map<RequestMetricKey, int64_t> duration_count_;
  std::map<RequestMetricKey, std::vector<int64_t>> duration_buckets_;
  std::map<ErrorMetricKey, int64_t> errors_total_;
  const std::vector<double> duration_bounds_seconds_{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0};
};

static std::string grpc_status_name(grpc::StatusCode code) {
  switch (code) {
    case grpc::StatusCode::OK:
      return "OK";
    case grpc::StatusCode::CANCELLED:
      return "CANCELLED";
    case grpc::StatusCode::UNKNOWN:
      return "UNKNOWN";
    case grpc::StatusCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case grpc::StatusCode::NOT_FOUND:
      return "NOT_FOUND";
    case grpc::StatusCode::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case grpc::StatusCode::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case grpc::StatusCode::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case grpc::StatusCode::ABORTED:
      return "ABORTED";
    case grpc::StatusCode::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case grpc::StatusCode::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case grpc::StatusCode::INTERNAL:
      return "INTERNAL";
    case grpc::StatusCode::UNAVAILABLE:
      return "UNAVAILABLE";
    case grpc::StatusCode::DATA_LOSS:
      return "DATA_LOSS";
    case grpc::StatusCode::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    default:
      return "UNKNOWN";
  }
}

class MetricsHttpServer {
 public:
  MetricsHttpServer(std::string host, int port, const MetricsRegistry& registry)
      : host_(std::move(host)), port_(port), registry_(registry) {
    if (port_ > 0) {
      server_thread_ = std::thread([this]() { Run(); });
    }
  }

  ~MetricsHttpServer() {
    stop_.store(true);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
  }

 private:
  void Run() {
    try {
      listen_fd_ = CreateListenSocket();
      if (listen_fd_ < 0) {
        log_error("metrics_server_failed", {{"error", "unable to bind metrics socket"}});
        return;
      }
      log_line("info", "metrics_server_started", {
                                                {"address", host_ + ":" + std::to_string(port_)},
                                            });
      while (!stop_.load()) {
        sockaddr_storage client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
          if (stop_.load()) {
            break;
          }
          if (errno == EINTR) {
            continue;
          }
          continue;
        }
        HandleClient(client_fd);
      }
    } catch (const std::exception& ex) {
      log_error("metrics_server_failed", {{"error", ex.what()}});
    }
  }

  int CreateListenSocket() const {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* result = nullptr;
    auto port_str = std::to_string(port_);
    if (::getaddrinfo(host_.empty() ? nullptr : host_.c_str(), port_str.c_str(), &hints, &result) != 0) {
      return -1;
    }

    int fd = -1;
    for (auto* ai = result; ai != nullptr; ai = ai->ai_next) {
      fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd < 0) {
        continue;
      }
      int yes = 1;
      ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
      if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 16) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    ::freeaddrinfo(result);
    return fd;
  }

  void HandleClient(int client_fd) const {
    std::string request;
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos) {
      ssize_t n = ::recv(client_fd, buffer, sizeof(buffer), 0);
      if (n <= 0) {
        ::close(client_fd);
        return;
      }
      request.append(buffer, static_cast<size_t>(n));
      if (request.size() > 8192) {
        break;
      }
    }

    std::istringstream first_line_stream(request);
    std::string method;
    std::string path;
    std::string version;
    first_line_stream >> method >> path >> version;

    std::string body;
    std::string status_line = "HTTP/1.1 404 Not Found\r\n";
    std::string content_type = "text/plain; version=0.0.4\r\n";

    if (method == "GET" && path == "/metrics") {
      body = registry_.Render();
      status_line = "HTTP/1.1 200 OK\r\n";
    } else if (method == "GET" && path == "/health") {
      body = "ok\n";
      status_line = "HTTP/1.1 200 OK\r\n";
      content_type = "text/plain\r\n";
    } else if (method != "GET") {
      body = "method not allowed\n";
      status_line = "HTTP/1.1 405 Method Not Allowed\r\n";
      content_type = "text/plain\r\n";
    } else {
      body = "not found\n";
    }

    std::ostringstream response;
    response << status_line;
    response << "Content-Type: " << content_type;
    response << "Content-Length: " << body.size() << "\r\n";
    response << "Connection: close\r\n\r\n";
    response << body;
    auto payload = response.str();
    ::send(client_fd, payload.data(), payload.size(), 0);
    ::close(client_fd);
  }

  std::string host_;
  int port_;
  const MetricsRegistry& registry_;
  mutable std::atomic<bool> stop_{false};
  mutable int listen_fd_{-1};
  std::thread server_thread_;
};

static Ort::Value create_ort_tensor(const ortserver::v1::Tensor& tensor, ONNXTensorElementDataType expected_type,
                                    InputStorage& storage, const Ort::MemoryInfo& memory_info) {
  switch (expected_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      storage.fp32 = repeated_to_vector<float>(tensor.fp32_data());
      auto shape = resolve_shape(repeated_to_vector<int64_t>(tensor.shape()), storage.fp32.size());
      return Ort::Value::CreateTensor<float>(memory_info, storage.fp32.data(), storage.fp32.size(), shape.data(),
                                             shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      if (!tensor.raw_data().empty()) {
        auto raw = bytes_to_vector<uint16_t>(tensor.raw_data());
        storage.fp16.resize(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
          storage.fp16[i] = Ort::Float16_t::FromBits(raw[i]);
        }
      } else {
        storage.fp16.resize(tensor.fp32_data_size());
        for (int i = 0; i < tensor.fp32_data_size(); ++i) {
          storage.fp16[static_cast<size_t>(i)] = Ort::Float16_t(tensor.fp32_data(i));
        }
      }
      auto shape = resolve_shape(repeated_to_vector<int64_t>(tensor.shape()), storage.fp16.size());
      return Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, storage.fp16.data(), storage.fp16.size(),
                                                      shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      storage.int32 = repeated_to_vector<int32_t>(tensor.int32_data());
      auto shape = resolve_shape(repeated_to_vector<int64_t>(tensor.shape()), storage.int32.size());
      return Ort::Value::CreateTensor<int32_t>(memory_info, storage.int32.data(), storage.int32.size(), shape.data(),
                                               shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      storage.int64 = repeated_to_vector<int64_t>(tensor.int64_data());
      auto shape = resolve_shape(repeated_to_vector<int64_t>(tensor.shape()), storage.int64.size());
      return Ort::Value::CreateTensor<int64_t>(memory_info, storage.int64.data(), storage.int64.size(), shape.data(),
                                               shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      storage.uint8.assign(tensor.raw_data().begin(), tensor.raw_data().end());
      auto shape = resolve_shape(repeated_to_vector<int64_t>(tensor.shape()), storage.uint8.size());
      return Ort::Value::CreateTensor<uint8_t>(memory_info, storage.uint8.data(), storage.uint8.size(), shape.data(),
                                               shape.size());
    }
    default:
      throw std::runtime_error("unsupported model input type");
  }
}

static ortserver::v1::Tensor ort_value_to_tensor(const std::string& name, const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  auto dtype = info.GetElementType();
  auto count = static_cast<size_t>(info.GetElementCount());

  ortserver::v1::Tensor tensor;
  tensor.set_name(name);
  for (int64_t dim : shape) {
    tensor.add_shape(dim);
  }

  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      tensor.set_datatype("FP32");
      const float* data = value.GetTensorData<float>();
      tensor.mutable_fp32_data()->Reserve(static_cast<int>(count));
      for (size_t i = 0; i < count; ++i) {
        tensor.add_fp32_data(data[i]);
      }
      return tensor;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      tensor.set_datatype("FP16");
      const Ort::Float16_t* data = value.GetTensorData<Ort::Float16_t>();
      std::string raw;
      raw.resize(count * sizeof(uint16_t));
      for (size_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &data[i], sizeof(bits));
        raw[i * 2] = static_cast<char>(bits & 0xFF);
        raw[i * 2 + 1] = static_cast<char>((bits >> 8) & 0xFF);
      }
      tensor.set_raw_data(raw);
      return tensor;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      tensor.set_datatype("INT32");
      const int32_t* data = value.GetTensorData<int32_t>();
      tensor.mutable_int32_data()->Reserve(static_cast<int>(count));
      for (size_t i = 0; i < count; ++i) {
        tensor.add_int32_data(data[i]);
      }
      return tensor;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      tensor.set_datatype("INT64");
      const int64_t* data = value.GetTensorData<int64_t>();
      tensor.mutable_int64_data()->Reserve(static_cast<int>(count));
      for (size_t i = 0; i < count; ++i) {
        tensor.add_int64_data(data[i]);
      }
      return tensor;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      tensor.set_datatype("UINT8");
      const uint8_t* data = value.GetTensorData<uint8_t>();
      std::string raw;
      raw.resize(count);
      for (size_t i = 0; i < count; ++i) {
        raw[i] = static_cast<char>(data[i]);
      }
      tensor.set_raw_data(raw);
      return tensor;
    }
    default:
      throw std::runtime_error("unsupported output type");
  }
}

class ModelStore {
 public:
  ModelStore(Ort::Env& env, const fs::path& model_root, int intra_threads, int inter_threads,
             int reload_interval_seconds)
      : env_(env), model_root_(model_root), intra_threads_(intra_threads), inter_threads_(inter_threads),
        reload_interval_seconds_(reload_interval_seconds) {
    ReloadOnce(true);
    if (reload_interval_seconds_ > 0) {
      reload_thread_ = std::thread([this]() { ReloadLoop(); });
    }
  }

  ~ModelStore() {
    stop_.store(true);
    reload_cv_.notify_all();
    if (reload_thread_.joinable()) {
      reload_thread_.join();
    }
  }

  std::shared_ptr<const LoadedModel> Get(const std::string& name,
                                         std::optional<int64_t> version = std::nullopt) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = versions_.find(name);
    if (it == versions_.end()) {
      throw std::runtime_error("unknown model: " + name);
    }
    if (version.has_value() && it->second->version != *version) {
      throw std::runtime_error("unknown model version: " + name + "/" + std::to_string(*version));
    }
    return it->second;
  }

 private:
  void ReloadLoop() {
    while (!stop_.load()) {
      std::unique_lock<std::mutex> lock(reload_mutex_);
      reload_cv_.wait_for(lock, std::chrono::seconds(reload_interval_seconds_),
                          [this]() { return stop_.load(); });
      if (stop_.load()) {
        break;
      }
      try {
        ReloadOnce(false);
      } catch (const std::exception& ex) {
        log_error("reload_failed", {{"error", ex.what()}});
      }
    }
  }

  std::shared_ptr<LoadedModel> LoadModel(const fs::path& model_path) const {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(intra_threads_);
    options.SetInterOpNumThreads(inter_threads_);
    options.DisableCpuMemArena();
    options.DisableMemPattern();

    // ORT keeps a process-wide cache of pre-packed operator weights (e.g.
    // transposed GEMM matrices) shared across all sessions in the same Env.
    // The cache key is based on weight content, so when a model version changes
    // the old entries are never evicted — they accumulate indefinitely, which
    // causes the "large RSS growth on first few reloads, then minor" pattern.
    //
    // Disabling weight sharing means each session owns its pre-packed weights
    // privately and they are freed exactly when the session is destroyed,
    // giving us deterministic memory reclamation on every version change.
    options.AddConfigEntry("session.use_env_allocators", "0");
    options.DisablePerSessionThreads();
    options.AddConfigEntry("session.disable_prepacking", "1");

    // Allocate with a custom deleter so malloc_trim(0) is called immediately
    // after the LoadedModel (and its Ort::Session) is destroyed, regardless of
    // which thread drops the last shared_ptr reference.  This ensures freed ORT
    // arena pages are returned to the OS even when an in-flight inference holds
    // the shared_ptr past the ReloadOnce swap point.
    auto* raw = new LoadedModel();
    std::shared_ptr<LoadedModel> model(raw, [](LoadedModel* p) {
      delete p;           // Ort::Session destructor frees arena back to glibc
      ::malloc_trim(0);   // glibc returns those free pages to the OS
    });
    auto [name, version] = parse_model_name_version(model_path);
    model->name = name;
    model->version = version;
    model->model_path = model_path;
    model->session = Ort::Session(env_, model_path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = model->session.GetInputCount();
    const size_t output_count = model->session.GetOutputCount();
    model->input_names.reserve(input_count);
    model->input_types.reserve(input_count);
    model->output_names.reserve(output_count);
    model->output_types.reserve(output_count);

    for (size_t i = 0; i < input_count; ++i) {
      auto name_ptr = model->session.GetInputNameAllocated(i, allocator);
      model->input_names.emplace_back(name_ptr.get());
      auto type_info = model->session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      model->input_types.push_back(tensor_info.GetElementType());
    }

    for (size_t i = 0; i < output_count; ++i) {
      auto name_ptr = model->session.GetOutputNameAllocated(i, allocator);
      model->output_names.emplace_back(name_ptr.get());
      auto type_info = model->session.GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      model->output_types.push_back(tensor_info.GetElementType());
    }

    return model;
  }

  void ReloadOnce(bool initial_load) {
    const int64_t rss_before = current_rss_bytes();
    auto discovered = discover_model_paths(model_root_);
    std::map<std::string, fs::path> latest_paths;
    std::map<std::string, int64_t> latest_versions;

    for (const auto& model_path : discovered) {
      auto [name, version] = parse_model_name_version(model_path);
      auto it = latest_versions.find(name);
      if (it == latest_versions.end() || version > it->second) {
        latest_versions[name] = version;
        latest_paths[name] = model_path;
      }
    }

    if (initial_load && latest_paths.empty()) {
      throw std::runtime_error("no ONNX models found under " + model_root_.string());
    }

    // Take a snapshot of current versions to decide what needs (re)loading.
    // Use a local scope so we release these shared_ptrs before loading new
    // models — holding them across LoadModel() doubles peak RSS unnecessarily.
    std::map<std::string, std::shared_ptr<LoadedModel>> current_versions;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_versions = versions_;
    }

    if (!initial_load && current_versions.size() == latest_paths.size()) {
      bool unchanged = true;
      for (const auto& [name, model_path] : latest_paths) {
        auto it = current_versions.find(name);
        if (it == current_versions.end() || it->second->version != latest_versions[name] ||
            it->second->model_path != model_path) {
          unchanged = false;
          break;
        }
      }
      if (unchanged) {
        return;
      }
    }

    std::map<std::string, std::shared_ptr<LoadedModel>> new_versions;
    for (const auto& [name, model_path] : latest_paths) {
      auto it = current_versions.find(name);
      if (it != current_versions.end() && it->second->version == latest_versions[name] &&
          it->second->model_path == model_path) {
        // Reuse the existing session — move the shared_ptr so current_versions
        // no longer holds a reference to it, allowing the old map to be freed
        // as soon as possible rather than pinning sessions until end-of-function.
        new_versions[name] = std::move(it->second);
        continue;
      }
      auto model = LoadModel(model_path);
      new_versions[name] = std::move(model);
    }

    // Release remaining entries in current_versions (models that were replaced
    // or removed) *before* the swap so that, when the old sessions are finally
    // evicted from versions_ below, their refcount can drop to zero immediately
    // instead of being kept alive by this local copy until end-of-function.
    current_versions.clear();

    std::map<std::string, int64_t> previous_versions;
    std::map<std::string, std::shared_ptr<LoadedModel>> stale_versions;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& [name, model] : versions_) {
        previous_versions[name] = model->version;
      }
      versions_.swap(new_versions);
      // new_versions now holds the old/stale sessions. Move them out of the
      // lock so we can destroy them (potentially blocking on ORT teardown)
      // without holding mutex_, keeping the critical section short.
      stale_versions = std::move(new_versions);
    }

    // Destroy old sessions outside the lock.  This is where ORT frees model
    // weights, thread pools, and arena buffers — it can be slow for large
    // models, so we must not block Get() callers during this teardown.
    stale_versions.clear();

    // ORT's arena allocator does a single large mmap and suballocates from it.
    // When the session is destroyed the arena is freed back to glibc, but
    // glibc holds those pages in its free list rather than returning them to
    // the OS — so RSS stays elevated even though the memory is logically free.
    // malloc_trim(0) forces glibc to release any free pages at the top of the
    // heap and to return any free mmap'd regions, which is what actually brings
    // RSS down after a version change.
    ::malloc_trim(0);

    const int64_t rss_after = current_rss_bytes();
    log_line("info", "reload_rss", {
                                      {"initial_load", initial_load ? "true" : "false"},
                                      {"rss_before_bytes", std::to_string(rss_before)},
                                      {"rss_after_bytes", std::to_string(rss_after)},
                                      {"rss_delta_bytes", std::to_string(rss_after - rss_before)},
                                      {"loaded_models", std::to_string(versions_.size())},
                                  });

    if (!initial_load) {
      for (const auto& [name, model] : versions_) {
        auto prev = previous_versions.find(name);
        if (prev == previous_versions.end()) {
          log_line("info", "model_loaded", {
                                           {"model", model->name},
                                           {"version", std::to_string(model->version)},
                                           {"path", model->model_path.string()},
                                       });
        } else if (prev->second != model->version) {
          log_line("info", "model_version_changed", {
                                                       {"model", name},
                                                       {"from", std::to_string(prev->second)},
                                                       {"to", std::to_string(model->version)},
                                                   });
          log_line("info", "model_loaded", {
                                           {"model", model->name},
                                           {"version", std::to_string(model->version)},
                                           {"path", model->model_path.string()},
                                       });
        }
      }
      for (const auto& [name, version] : previous_versions) {
        if (!versions_.count(name)) {
          log_line("info", "model_family_removed", {{"model", name}});
        }
      }
    } else {
      for (const auto& [name, model] : versions_) {
        log_line("info", "model_loaded", {
                                         {"model", model->name},
                                         {"version", std::to_string(model->version)},
                                         {"path", model->model_path.string()},
                                     });
      }
    }
  }

  Ort::Env& env_;
  fs::path model_root_;
  int intra_threads_;
  int inter_threads_;
  int reload_interval_seconds_;
  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<LoadedModel>> versions_;
  std::atomic<bool> stop_{false};
  std::mutex reload_mutex_;
  std::condition_variable reload_cv_;
  std::thread reload_thread_;
};

class OrtServiceImpl final : public ortserver::v1::OrtService::Service {
 public:
  OrtServiceImpl(const ModelStore& store, MetricsRegistry& metrics, bool log_predict)
      : store_(store), metrics_(metrics), log_predict_(log_predict) {}

  grpc::Status Predict(grpc::ServerContext* context, const ortserver::v1::PredictRequest* request,
                       ortserver::v1::PredictResponse* response) override {
    auto started = Clock::now();
    auto decode_started = started;
    auto run_started = started;
    try {
      if (request->model_spec().name().empty()) {
        auto ended = Clock::now();
        record_predict("", "0", "400", started, started, started, ended, "invalid_argument");
        return grpc::Status(grpc::INVALID_ARGUMENT, "model_spec.name is required");
      }

      auto model = store_.Get(request->model_spec().name(), std::nullopt);
      const std::string model_key = model->name + "/" + std::to_string(model->version);
      const std::string version_key = std::to_string(model->version);
      auto requested_signature = request->model_spec().signature_name();
      if (!requested_signature.empty() && requested_signature != "serving_default") {
        log_line("info", "ignore_signature", {
                                            {"model", model->name},
                                            {"version", std::to_string(model->version)},
                                            {"signature", requested_signature},
                                        });
      }

      if (request->inputs().empty()) {
        auto ended = Clock::now();
        record_predict(model->name, version_key, "400", started, started, started, ended, "invalid_argument");
        return grpc::Status(grpc::INVALID_ARGUMENT, "inputs are required");
      }

      std::vector<InputStorage> storages;
      storages.reserve(model->input_names.size());
      std::vector<Ort::Value> input_values;
      input_values.reserve(model->input_names.size());
      std::vector<const char*> input_names;
      input_names.reserve(model->input_names.size());

      if (model->input_names.size() == 1 && request->inputs_size() == 1) {
        const auto& input = request->inputs().begin()->second;
        storages.emplace_back();
        input_values.push_back(create_ort_tensor(input, model->input_types[0], storages.back(),
                                                 Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)));
        input_names.push_back(model->input_names[0].c_str());
      } else if (request->inputs_size() == static_cast<int>(model->input_names.size())) {
        std::unordered_map<std::string, const ortserver::v1::Tensor*> resolved;
        for (const auto& kv : request->inputs()) {
          resolved.emplace(kv.first, &kv.second);
        }
        for (size_t i = 0; i < model->input_names.size(); ++i) {
          auto it = resolved.find(model->input_names[i]);
          if (it == resolved.end()) {
            auto ended = Clock::now();
            record_predict(model->name, version_key, "400", started, started, started, ended, "invalid_argument");
            return grpc::Status(grpc::INVALID_ARGUMENT, "missing request input: " + model->input_names[i]);
          }
          storages.emplace_back();
          input_values.push_back(create_ort_tensor(*it->second, model->input_types[i], storages.back(),
                                                   Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)));
          input_names.push_back(model->input_names[i].c_str());
        }
      } else {
        auto ended = Clock::now();
        record_predict(model->name, version_key, "400", started, started, started, ended, "invalid_argument");
        return grpc::Status(grpc::INVALID_ARGUMENT, "unable to map request inputs to model inputs");
      }

      auto decode_done = Clock::now();
      run_started = decode_done;

      std::vector<const char*> output_names;
      output_names.reserve(model->output_names.size());
      for (const auto& out_name : model->output_names) {
        output_names.push_back(out_name.c_str());
      }

      auto ort_outputs = model->session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                                           input_values.size(), output_names.data(), output_names.size());
      auto run_done = Clock::now();

      auto* response_model_spec = response->mutable_model_spec();
      response_model_spec->set_name(model->name);
      response_model_spec->set_version(std::to_string(model->version));
      response_model_spec->set_signature_name(request->model_spec().signature_name());

      for (size_t i = 0; i < ort_outputs.size(); ++i) {
        (*response->mutable_outputs())[output_names[i]] = ort_value_to_tensor(output_names[i], ort_outputs[i]);
      }

      auto ended = Clock::now();
      record_predict(model_key, version_key, "200", started, decode_done, run_done, ended, "");
      return grpc::Status::OK;
    } catch (const std::exception& ex) {
      auto ended = Clock::now();
      const std::string model = request->model_spec().name().empty() ? "unknown" : request->model_spec().name();
      record_predict(model, "latest", "500",
                     started, decode_started, run_started, ended, "internal");
      return grpc::Status(grpc::INTERNAL, ex.what());
    }
  }

  grpc::Status Health(grpc::ServerContext* context, const ortserver::v1::HealthRequest* request,
                      ortserver::v1::HealthResponse* response) override {
    response->set_status("ok");
    return grpc::Status::OK;
  }

 private:
  void record_predict(const std::string& model, const std::string& version, const std::string& status,
                      const Clock::time_point& started, const Clock::time_point& decode_done,
                      const Clock::time_point& run_done, const Clock::time_point& ended,
                      const std::string& error) const {
    auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_done - started).count();
    auto run_us = std::chrono::duration_cast<std::chrono::microseconds>(run_done - decode_done).count();
    auto encode_us = std::chrono::duration_cast<std::chrono::microseconds>(ended - run_done).count();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(ended - started).count();
    metrics_.ObserveRequest(model, version, status, total_us, error);
    if (log_predict_) {
      log_line("info", "predict", {
                                  {"model", model},
                                  {"status", status},
                                  {"decode_us", std::to_string(decode_us)},
                                  {"run_us", std::to_string(run_us)},
                                  {"encode_us", std::to_string(encode_us)},
                                  {"duration_us", std::to_string(total_us)},
                                  {"error", error},
                              });
    }
  }

  const ModelStore& store_;
  MetricsRegistry& metrics_;
  bool log_predict_;
};

struct Args {
  fs::path model_root = fs::path("../onnx_models");
  std::string host = "0.0.0.0";
  int port = 18500;
  std::string metrics_host = "0.0.0.0";
  int metrics_port = 9090;
  int workers = 4;
  int max_message_mb = 16;
  int intra_threads = 0;
  int inter_threads = 1;
  int reload_interval_seconds = 60;
  bool log_predict = false;
};

static const char* env_or_null(const char* name) {
  return std::getenv(name);
}

static std::optional<std::string> env_string(const char* name) {
  if (const char* value = env_or_null(name); value != nullptr && value[0] != '\0') {
    return std::string(value);
  }
  return std::nullopt;
}

static std::optional<int> env_int(const char* name) {
  if (auto value = env_string(name)) {
    return std::stoi(*value);
  }
  return std::nullopt;
}

static std::optional<bool> env_bool(const char* name) {
  if (auto value = env_string(name)) {
    std::string lowered = *value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
      return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
      return false;
    }
    throw std::runtime_error("invalid boolean value for " + std::string(name) + ": " + *value);
  }
  return std::nullopt;
}

static Args parse_args(int argc, char** argv) {
  Args args;
  if (auto value = env_string("ORTSERVER_MODEL_ROOT")) {
    args.model_root = *value;
  }
  if (auto value = env_string("ORTSERVER_HOST")) {
    args.host = *value;
  }
  if (auto value = env_int("ORTSERVER_PORT")) {
    args.port = *value;
  }
  if (auto value = env_string("ORTSERVER_METRICS_HOST")) {
    args.metrics_host = *value;
  }
  if (auto value = env_int("ORTSERVER_METRICS_PORT")) {
    args.metrics_port = *value;
  }
  if (auto value = env_int("ORTSERVER_WORKERS")) {
    args.workers = *value;
  }
  if (auto value = env_int("ORTSERVER_MAX_MESSAGE_MB")) {
    args.max_message_mb = *value;
  }
  if (auto value = env_int("ORTSERVER_INTRA_THREADS")) {
    args.intra_threads = *value;
  }
  if (auto value = env_int("ORTSERVER_INTER_THREADS")) {
    args.inter_threads = *value;
  }
  if (auto value = env_int("ORTSERVER_RELOAD_INTERVAL_SECONDS")) {
    args.reload_interval_seconds = *value;
  }
  if (auto value = env_bool("ORTSERVER_LOG_PREDICT")) {
    args.log_predict = *value;
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto consume = [&](std::string_view flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(flag));
      }
      return argv[++i];
    };
    if (arg == "--model-root") {
      args.model_root = consume(arg);
    } else if (arg == "--host") {
      args.host = consume(arg);
    } else if (arg == "--port") {
      args.port = std::stoi(consume(arg));
    } else if (arg == "--metrics-host") {
      args.metrics_host = consume(arg);
    } else if (arg == "--metrics-port") {
      args.metrics_port = std::stoi(consume(arg));
    } else if (arg == "--workers") {
      args.workers = std::stoi(consume(arg));
    } else if (arg == "--max-message-mb") {
      args.max_message_mb = std::stoi(consume(arg));
    } else if (arg == "--intra-threads") {
      args.intra_threads = std::stoi(consume(arg));
    } else if (arg == "--inter-threads") {
      args.inter_threads = std::stoi(consume(arg));
    } else if (arg == "--reload-interval-seconds") {
      args.reload_interval_seconds = std::stoi(consume(arg));
    } else if (arg == "--log-predict") {
      args.log_predict = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);
    if (args.intra_threads <= 0) {
      args.intra_threads = default_intra_threads();
    }

    // Use a global thread pool in the Env so sessions created with
    // DisablePerSessionThreads() share a single pool rather than each spawning
    // and leaking their own intra/inter-op threads on every version reload.
    OrtThreadingOptions* tp_opts_raw = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateThreadingOptions(&tp_opts_raw));
    Ort::ThrowOnError(Ort::GetApi().SetGlobalIntraOpNumThreads(tp_opts_raw, args.intra_threads));
    Ort::ThrowOnError(Ort::GetApi().SetGlobalInterOpNumThreads(tp_opts_raw, args.inter_threads));
    Ort::Env env(tp_opts_raw, ORT_LOGGING_LEVEL_ERROR, "ort_grpc");
    Ort::GetApi().ReleaseThreadingOptions(tp_opts_raw);

    ModelStore store(env, args.model_root, args.intra_threads, args.inter_threads, args.reload_interval_seconds);
    MetricsRegistry metrics;
    MetricsHttpServer metrics_server(args.metrics_host, args.metrics_port, metrics);
    OrtServiceImpl service(store, metrics, args.log_predict);

    std::string server_address = args.host + ":" + std::to_string(args.port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(args.max_message_mb * 1024 * 1024);
    builder.SetMaxSendMessageSize(args.max_message_mb * 1024 * 1024);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    log_line("info", "server_started", {
                                         {"address", server_address},
                                         {"metrics_address", args.metrics_host + ":" + std::to_string(args.metrics_port)},
                                         {"reload_interval_seconds", std::to_string(args.reload_interval_seconds)},
                                         {"max_message_mb", std::to_string(args.max_message_mb)},
                                         {"log_predict", args.log_predict ? "true" : "false"},
                                     });
    server->Wait();
  } catch (const std::exception& ex) {
    log_error("startup_failed", {{"error", ex.what()}});
    return 1;
  }
  return 0;
}