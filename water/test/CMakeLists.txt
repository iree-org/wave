configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WATER_TEST_DEPENDS
        FileCheck count not
        water-opt
        )
if(WATER_ENABLE_PYTHON)
  list(APPEND WATER_TEST_DEPENDS
       WaterPythonModules
)
endif()

add_lit_testsuite(check-water "Running the water regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${WATER_TEST_DEPENDS}
        )
set_target_properties(check-water PROPERTIES FOLDER "Tests")

add_subdirectory(lib)
