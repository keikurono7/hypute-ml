cmake_minimum_required(VERSION 3.12)
project(HyputeML CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread -funroll-loops")

link_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(hypute_ml_eval     evaluation/main.cpp)
add_executable(hypute_ml_bench    benchmarks/inference/benchmark.cpp)
add_executable(generate_weights   tools/generate_weights.cpp)
add_executable(sentiment_eval     examples/sentiment/eval.cpp)
add_executable(lm_generate        examples/lm/generate.cpp)
add_executable(chat               examples/chatbot/chat.cpp)

foreach(t hypute_ml_eval hypute_ml_bench generate_weights sentiment_eval)
    target_link_libraries(${t} hypute_ml pthread)
endforeach()

foreach(t lm_generate chat)
    target_link_libraries(${t} hypute_ml_lm pthread)
endforeach()
