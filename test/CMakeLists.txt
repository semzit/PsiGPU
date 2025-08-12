include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(PsiGPU_test
  test_kernels.cpp
)

target_link_libraries(PsiGPU_test PRIVATE PsiGPULib gtest gtest_main)

target_include_directories(PsiGPU_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

find_package(GTest)

target_link_libraries(
  PsiGPU_test
  PRIVATE PsiGPULib
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(PsiGPU_test)
