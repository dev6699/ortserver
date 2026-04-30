cmake_minimum_required(VERSION 3.19)
project(ortserver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Protobuf_INCLUDE_DIR /root/.local/include CACHE PATH "protobuf include dir" FORCE)
set(Protobuf_LIBRARY /root/.local/lib/libprotobuf.so CACHE FILEPATH "protobuf library" FORCE)
set(Protobuf_PROTOC_EXECUTABLE /root/.local/bin/protoc CACHE FILEPATH "protobuf compiler" FORCE)
set(gRPC_CPP_PLUGIN_EXECUTABLE /root/.local/bin/grpc_cpp_plugin CACHE FILEPATH "gRPC C++ plugin" FORCE)

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

set(PROTOC_EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}")
set(GRPC_CPP_PLUGIN_EXECUTABLE "${gRPC_CPP_PLUGIN_EXECUTABLE}")

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the extracted ONNX Runtime release root")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<ver>")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain include/onnxruntime_cxx_api.h: ${ONNXRUNTIME_ROOT}")
endif()

set(ORT_PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(GEN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(GLOB_RECURSE ORT_PROTO_FILES RELATIVE "${ORT_PROTO_ROOT}" "${ORT_PROTO_ROOT}/*.proto")

set(ORT_GENERATED_SRCS)
set(ORT_GENERATED_HDRS)

foreach(proto_file IN LISTS ORT_PROTO_FILES)
  get_filename_component(ort_proto_dir "${proto_file}" DIRECTORY)
  get_filename_component(ort_proto_name "${proto_file}" NAME_WE)
  set(out_cc "${GEN_ROOT}/${ort_proto_name}.pb.cc")
  set(out_h "${GEN_ROOT}/${ort_proto_name}.pb.h")
  add_custom_command(
    OUTPUT "${out_cc}" "${out_h}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${GEN_ROOT}"
    COMMAND "${PROTOC_EXECUTABLE}"
      --proto_path=${ORT_PROTO_ROOT}
      --cpp_out=${GEN_ROOT}
      ${ORT_PROTO_ROOT}/${proto_file}
    DEPENDS "${ORT_PROTO_ROOT}/${proto_file}"
    COMMENT "Generating protobuf for ${proto_file}"
    VERBATIM
  )
  list(APPEND ORT_GENERATED_SRCS "${out_cc}")
  list(APPEND ORT_GENERATED_HDRS "${out_h}")

  set(out_cc_grpc "${GEN_ROOT}/${ort_proto_name}.grpc.pb.cc")
  set(out_h_grpc "${GEN_ROOT}/${ort_proto_name}.grpc.pb.h")
  add_custom_command(
    OUTPUT "${out_cc_grpc}" "${out_h_grpc}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${GEN_ROOT}"
    COMMAND "${PROTOC_EXECUTABLE}"
      --proto_path=${ORT_PROTO_ROOT}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
      --grpc_out=${GEN_ROOT}
      ${ORT_PROTO_ROOT}/${proto_file}
    DEPENDS "${ORT_PROTO_ROOT}/${proto_file}"
    COMMENT "Generating gRPC stubs for ${proto_file}"
    VERBATIM
  )
  list(APPEND ORT_GENERATED_SRCS "${out_cc_grpc}")
  list(APPEND ORT_GENERATED_HDRS "${out_h_grpc}")
endforeach()

add_library(ortserver_protos STATIC ${ORT_GENERATED_SRCS})
target_include_directories(ortserver_protos PUBLIC "${GEN_ROOT}" "${ORT_PROTO_ROOT}")
target_link_libraries(ortserver_protos PUBLIC gRPC::grpc++ protobuf::libprotobuf)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_executable(ortserver main.cc)
add_dependencies(ortserver ortserver_protos)
target_include_directories(ortserver PRIVATE "${GEN_ROOT}" "${ORT_PROTO_ROOT}" "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(ortserver PRIVATE ortserver_protos onnxruntime Threads::Threads protobuf::libprotobuf)
target_link_options(ortserver PRIVATE -static-libstdc++ -static-libgcc)
