include_directories(SYSTEM ${DUNE_XT_COMMON_TEST_DIR}/gtest)

enable_testing()

begin_testcases()

end_testcases()

foreach(test ${dxt_test_binaries})
  if(${test} MATCHES alberta_1d)
    add_dune_alberta_flags(GRIDDIM 1 ${test})
  elseif(${test} MATCHES alberta_2d)
    add_dune_alberta_flags(GRIDDIM 2 ${test})
  elseif(${test} MATCHES alberta_3d)
    add_dune_alberta_flags(GRIDDIM 3 ${test})
  endif()
endforeach()

