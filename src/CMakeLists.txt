add_executable(${PROJECT_NAME}
  cll.cpp
  cornelius.cpp
  eoChiral.cpp
  eoHadron.cpp
  eos.cpp
  eos_hotqcd.cpp
  fast.cpp
  fld.cpp
  hdo.cpp
  icTrento.cpp
  rmn.cpp
  trancoeff.cpp
  main.cpp
)

#target_link_libraries(${PROJECT_NAME} ${ROOT_LIB})
target_link_libraries(${PROJECT_NAME} ${HDF5_LIBRARIES})
install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME})
#install(TARGETS ${PROJECT_NAME} DESTINATION bin)
