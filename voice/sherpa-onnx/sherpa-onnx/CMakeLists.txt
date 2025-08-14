add_subdirectory(csrc)

add_subdirectory(voice)
if(SHERPA_ONNX_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if(SHERPA_ONNX_ENABLE_JNI)
  add_subdirectory(jni)
endif()

if(SHERPA_ONNX_ENABLE_C_API)
  add_subdirectory(c-api)
endif()
